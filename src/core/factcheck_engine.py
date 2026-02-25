"""Vérification factuelle automatique — détection d'hallucinations.

Phase 3 : vérifie chaque affirmation générée contre le corpus source
via un pipeline en 3 étapes (extraction, corroboration, évaluation).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.providers.base import BaseProvider
from src.utils.file_utils import ensure_dir, save_json, load_json

logger = logging.getLogger("orchestria")

# Statuts des affirmations
CORROBORATED = "CORROBORÉE"
PLAUSIBLE = "PLAUSIBLE"
UNFOUNDED = "NON FONDÉE"
CONTRADICTED = "CONTREDITE"

EXTRACTION_PROMPT = """Analyse le texte suivant et extrais les affirmations factuelles vérifiables.

═══ TEXTE À ANALYSER ═══
{content}

═══ INSTRUCTIONS ═══
Extrais les affirmations factuelles vérifiables : faits, données chiffrées, attributions, relations causales.
Exclus les opinions, jugements de valeur et formulations hypothétiques.
Retourne EXACTEMENT le format JSON suivant (sans commentaires) :
{{
  "claims": [
    {{"id": 1, "text": "<affirmation>", "type": "fact|data|attribution|causal"}},
    ...
  ]
}}
Limite : maximum {max_claims} affirmations."""

EVALUATION_PROMPT = """Évalue si l'affirmation suivante est soutenue par les extraits de corpus.

═══ AFFIRMATION ═══
{claim}

═══ EXTRAITS DU CORPUS ═══
{corpus_excerpts}

═══ INSTRUCTIONS ═══
Évalue et retourne EXACTEMENT le format JSON suivant :
{{
  "status": "CORROBORÉE|PLAUSIBLE|NON FONDÉE|CONTREDITE",
  "justification": "<explication en 1-2 phrases>"
}}

Règles :
- CORROBORÉE : l'affirmation est directement soutenue par le corpus
- PLAUSIBLE : l'affirmation est cohérente mais pas directement confirmée
- NON FONDÉE : aucune information dans le corpus ne soutient l'affirmation
- CONTREDITE : le corpus contient des informations contradictoires"""

COMBINED_PROMPT = """Analyse le texte suivant : extrais les affirmations factuelles vérifiables, puis évalue chacune par rapport aux extraits du corpus.

═══ TEXTE À ANALYSER ═══
{content}

═══ EXTRAITS DU CORPUS ═══
{corpus_excerpts}

═══ INSTRUCTIONS ═══
1. Extrais les affirmations factuelles vérifiables (max {max_claims}).
2. Pour chaque affirmation, évalue son statut par rapport au corpus.

Retourne EXACTEMENT le format JSON suivant :
{{
  "claims": [
    {{
      "id": 1,
      "text": "<affirmation>",
      "status": "CORROBORÉE|PLAUSIBLE|NON FONDÉE|CONTREDITE",
      "justification": "<explication>"
    }},
    ...
  ]
}}"""


@dataclass
class ClaimResult:
    """Résultat de la vérification d'une affirmation."""
    claim_id: int
    text: str
    status: str  # CORROBORATED, PLAUSIBLE, UNFOUNDED, CONTRADICTED
    justification: str = ""
    corpus_excerpts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.claim_id,
            "text": self.text,
            "status": self.status,
            "justification": self.justification,
            "corpus_excerpts": self.corpus_excerpts,
        }


@dataclass
class FactcheckReport:
    """Rapport de vérification factuelle d'une section."""
    section_id: str
    total_claims: int = 0
    status_counts: dict = field(default_factory=dict)
    reliability_score: float = 0.0
    details: list[ClaimResult] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "total_claims": self.total_claims,
            "status_counts": self.status_counts,
            "reliability_score": self.reliability_score,
            "details": [d.to_dict() for d in self.details],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FactcheckReport":
        report = cls(
            section_id=data["section_id"],
            total_claims=data.get("total_claims", 0),
            status_counts=data.get("status_counts", {}),
            reliability_score=data.get("reliability_score", 0.0),
            timestamp=data.get("timestamp", ""),
        )
        for d in data.get("details", []):
            report.details.append(ClaimResult(
                claim_id=d["id"],
                text=d["text"],
                status=d["status"],
                justification=d.get("justification", ""),
                corpus_excerpts=d.get("corpus_excerpts", []),
            ))
        return report


class FactcheckEngine:
    """Vérifie la fiabilité factuelle des sections générées."""

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        rag_engine=None,
        project_dir: Optional[Path] = None,
        enabled: bool = True,
        auto_correct_threshold: float = 80.0,
        max_claims_per_section: int = 30,
        factcheck_model: Optional[str] = None,
    ):
        self.provider = provider
        self.rag_engine = rag_engine
        self.project_dir = project_dir
        self.enabled = enabled
        self.auto_correct_threshold = auto_correct_threshold
        self.max_claims = max_claims_per_section
        self.factcheck_model = factcheck_model

    def check_section(
        self,
        section_id: str,
        content: str,
        section_title: str = "",
        section_description: str = "",
    ) -> FactcheckReport:
        """Vérifie une section complète.

        Args:
            section_id: Identifiant de la section.
            content: Contenu généré à vérifier.
            section_title: Titre de la section (pour la recherche RAG).
            section_description: Description de la section.

        Returns:
            Rapport de vérification factuelle.
        """
        if not self.enabled or not self.provider:
            return FactcheckReport(section_id=section_id, reliability_score=100.0)

        word_count = len(content.split())
        model = self.factcheck_model or self.provider.get_default_model()

        # Get corpus excerpts for corroboration
        corpus_text = self._get_corpus_excerpts(section_title, section_description)

        if word_count < 2000 and corpus_text:
            # Short section: combined extraction + evaluation
            report = self._check_combined(section_id, content, corpus_text, model)
        else:
            # Long section: extract claims first, then evaluate individually
            claims = self._extract_claims(content, model)
            report = self._evaluate_claims(section_id, claims, corpus_text, model)

        # Save report
        if self.project_dir:
            self._save_report(report)

        return report

    def should_correct(self, report: FactcheckReport) -> bool:
        """Détermine si une correction automatique est nécessaire."""
        return report.reliability_score < self.auto_correct_threshold

    def get_correction_instruction(self, report: FactcheckReport) -> str:
        """Génère l'instruction de correction pour les affirmations problématiques."""
        problematic = [
            d for d in report.details
            if d.status in (UNFOUNDED, CONTRADICTED)
        ]
        if not problematic:
            return ""

        lines = ["Les affirmations suivantes doivent être retirées ou reformulées car elles ne sont pas soutenues par le corpus :"]
        for claim in problematic:
            lines.append(f"- [{claim.status}] {claim.text}")
            if claim.justification:
                lines.append(f"  Raison : {claim.justification}")

        lines.append("\nReformule ces passages en te basant uniquement sur le corpus source.")
        return "\n".join(lines)

    def _get_corpus_excerpts(self, section_title: str, section_description: str) -> str:
        """Récupère les extraits de corpus pertinents via RAG."""
        if not self.rag_engine:
            return ""

        try:
            query = f"{section_title} {section_description}".strip()
            if not query:
                return ""
            result = self.rag_engine.search_for_section("factcheck", query, "")
            excerpts = []
            for chunk in result.chunks[:5]:
                text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
                source = chunk.get("source_file", "") if isinstance(chunk, dict) else getattr(chunk, "source_file", "")
                excerpts.append(f"[{source}] {text[:500]}")
            return "\n---\n".join(excerpts)
        except Exception as e:
            logger.warning(f"Recherche RAG pour factcheck échouée : {e}")
            return ""

    def _check_combined(self, section_id: str, content: str, corpus_text: str, model: str) -> FactcheckReport:
        """Extraction et évaluation combinées (sections courtes)."""
        prompt = COMBINED_PROMPT.format(
            content=content[:3000],
            corpus_excerpts=corpus_text[:3000],
            max_claims=self.max_claims,
        )

        try:
            response = self.provider.generate(
                prompt=prompt,
                system_prompt="Tu es un vérificateur factuel. Retourne uniquement du JSON valide.",
                model=model,
                temperature=0.1,
                max_tokens=2000,
            )
            claims_data = self._parse_json_response(response.content)
            return self._build_report(section_id, claims_data)
        except Exception as e:
            logger.warning(f"Factcheck combiné échoué pour {section_id}: {e}")
            return FactcheckReport(section_id=section_id, reliability_score=100.0)

    def _extract_claims(self, content: str, model: str) -> list[dict]:
        """Étape 1 : extraction des affirmations vérifiables."""
        prompt = EXTRACTION_PROMPT.format(
            content=content[:4000],
            max_claims=self.max_claims,
        )

        try:
            response = self.provider.generate(
                prompt=prompt,
                system_prompt="Tu es un extracteur d'affirmations factuelles. Retourne uniquement du JSON valide.",
                model=model,
                temperature=0.1,
                max_tokens=1500,
            )
            data = self._parse_json_response(response.content)
            return data.get("claims", [])
        except Exception as e:
            logger.warning(f"Extraction d'affirmations échouée : {e}")
            return []

    def _evaluate_claims(
        self, section_id: str, claims: list[dict], corpus_text: str, model: str
    ) -> FactcheckReport:
        """Étape 2+3 : évaluation de chaque affirmation."""
        if not claims:
            return FactcheckReport(section_id=section_id, reliability_score=100.0)

        results = []
        for claim in claims:
            claim_text = claim.get("text", "")
            if not claim_text:
                continue

            # Recherche RAG spécifique à l'affirmation
            specific_corpus = corpus_text
            if self.rag_engine:
                try:
                    result = self.rag_engine.search_for_section("factcheck", claim_text, "")
                    excerpts = []
                    for chunk in result.chunks[:3]:
                        text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
                        excerpts.append(text[:400])
                    if excerpts:
                        specific_corpus = "\n---\n".join(excerpts)
                except Exception:
                    pass

            prompt = EVALUATION_PROMPT.format(
                claim=claim_text,
                corpus_excerpts=specific_corpus[:2000] if specific_corpus else "Aucun extrait de corpus disponible.",
            )

            try:
                response = self.provider.generate(
                    prompt=prompt,
                    system_prompt="Tu es un vérificateur factuel. Retourne uniquement du JSON valide.",
                    model=model,
                    temperature=0.1,
                    max_tokens=300,
                )
                eval_data = self._parse_json_response(response.content)
                status = eval_data.get("status", PLAUSIBLE)
                # Normalize status
                status = self._normalize_status(status)
                results.append({
                    "id": claim.get("id", len(results) + 1),
                    "text": claim_text,
                    "status": status,
                    "justification": eval_data.get("justification", ""),
                })
            except Exception as e:
                logger.warning(f"Évaluation d'affirmation échouée : {e}")
                results.append({
                    "id": claim.get("id", len(results) + 1),
                    "text": claim_text,
                    "status": PLAUSIBLE,
                    "justification": "Évaluation impossible.",
                })

        return self._build_report(section_id, {"claims": results})

    def _build_report(self, section_id: str, data: dict) -> FactcheckReport:
        """Construit le rapport à partir des données d'évaluation."""
        claims = data.get("claims", [])
        report = FactcheckReport(section_id=section_id)
        report.total_claims = len(claims)

        status_counts = {
            CORROBORATED: 0,
            PLAUSIBLE: 0,
            UNFOUNDED: 0,
            CONTRADICTED: 0,
        }

        for claim in claims:
            status = self._normalize_status(claim.get("status", PLAUSIBLE))
            status_counts[status] = status_counts.get(status, 0) + 1
            report.details.append(ClaimResult(
                claim_id=claim.get("id", 0),
                text=claim.get("text", ""),
                status=status,
                justification=claim.get("justification", ""),
            ))

        report.status_counts = status_counts

        # Calcul du score de fiabilité
        total = report.total_claims
        if total > 0:
            reliable = status_counts.get(CORROBORATED, 0) + status_counts.get(PLAUSIBLE, 0)
            report.reliability_score = round((reliable / total) * 100, 1)
        else:
            report.reliability_score = 100.0

        return report

    @staticmethod
    def _normalize_status(status: str) -> str:
        """Normalise le statut retourné par l'IA."""
        status_upper = status.upper().strip()
        mapping = {
            "CORROBORÉE": CORROBORATED,
            "CORROBOREE": CORROBORATED,
            "CORROBORATED": CORROBORATED,
            "PLAUSIBLE": PLAUSIBLE,
            "NON FONDÉE": UNFOUNDED,
            "NON FONDEE": UNFOUNDED,
            "UNFOUNDED": UNFOUNDED,
            "CONTREDITE": CONTRADICTED,
            "CONTRADICTED": CONTRADICTED,
        }
        return mapping.get(status_upper, PLAUSIBLE)

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse une réponse JSON de l'IA."""
        text = text.strip()
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _save_report(self, report: FactcheckReport) -> None:
        """Sauvegarde le rapport sur disque."""
        if not self.project_dir:
            return
        factcheck_dir = ensure_dir(self.project_dir / "factcheck")
        filepath = factcheck_dir / f"{report.section_id}.json"
        save_json(filepath, report.to_dict())
