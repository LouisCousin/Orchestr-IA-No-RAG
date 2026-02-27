"""Évaluation automatique de la qualité des sections générées.

Phase 3 : évalue chaque section sur 6 critères et produit un rapport
structuré avec score global pondéré.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.export_engine import detect_needs_source_markers
from src.core.plan_parser import PlanSection, NormalizedPlan
from src.providers.base import BaseProvider

logger = logging.getLogger("orchestria")

# Pondérations par défaut des critères
DEFAULT_WEIGHTS = {
    "plan_conformity": 1.0,
    "corpus_coverage": 1.5,
    "narrative_coherence": 0.8,
    "target_size": 0.5,
    "factual_reliability": 1.5,
    "source_traceability": 1.2,
}

EVALUATION_PROMPT = """Tu es un évaluateur de qualité documentaire. Évalue le contenu suivant selon les critères demandés.

═══ SECTION ÉVALUÉE ═══
Titre : {section_title}
Description : {section_description}

═══ CONTENU GÉNÉRÉ ═══
{content}

═══ CORPUS SOURCE (EXTRAITS) ═══
{corpus_summary}

═══ SECTIONS PRÉCÉDENTES (RÉSUMÉS) ═══
{previous_summaries}

═══ CRITÈRES D'ÉVALUATION ═══
Évalue chaque critère ci-dessous avec un score de 1 à 5 et une justification courte (1 phrase).

C1 — Conformité au plan : Le contenu correspond-il au titre et à la description de la section ?
C2 — Couverture du corpus : Le contenu s'appuie-t-il sur les blocs de corpus fournis ?
C3 — Cohérence narrative : Style, ton et vocabulaire homogènes avec les sections précédentes ?

Retourne EXACTEMENT le format JSON suivant (sans commentaires) :
{{
  "C1": {{"score": <1-5>, "justification": "<texte>"}},
  "C2": {{"score": <1-5>, "justification": "<texte>"}},
  "C3": {{"score": <1-5>, "justification": "<texte>"}}
}}
"""


@dataclass
class CriterionResult:
    """Résultat d'un critère d'évaluation."""
    criterion_id: str
    criterion_name: str
    score: float  # 1.0 to 5.0
    justification: str = ""
    weight: float = 1.0


@dataclass
class QualityReport:
    """Rapport de qualité d'une section."""
    section_id: str
    criteria: list[CriterionResult] = field(default_factory=list)
    global_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    needs_source_count: int = 0

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "criteria": [
                {
                    "id": c.criterion_id,
                    "name": c.criterion_name,
                    "score": c.score,
                    "justification": c.justification,
                    "weight": c.weight,
                }
                for c in self.criteria
            ],
            "global_score": self.global_score,
            "recommendations": self.recommendations,
            "needs_source_count": self.needs_source_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QualityReport":
        report = cls(
            section_id=data["section_id"],
            global_score=data.get("global_score", 0.0),
            recommendations=data.get("recommendations", []),
            needs_source_count=data.get("needs_source_count", 0),
        )
        for c in data.get("criteria", []):
            report.criteria.append(CriterionResult(
                criterion_id=c["id"],
                criterion_name=c["name"],
                score=c["score"],
                justification=c.get("justification", ""),
                weight=c.get("weight", 1.0),
            ))
        return report


class QualityEvaluator:
    """Évalue la qualité des sections générées sur 6 critères."""

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        weights: Optional[dict] = None,
        auto_refine_threshold: float = 3.0,
        evaluation_model: Optional[str] = None,
        enabled: bool = True,
    ):
        self.provider = provider
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.auto_refine_threshold = auto_refine_threshold
        self.evaluation_model = evaluation_model
        self.enabled = enabled

    def evaluate_section(
        self,
        section: PlanSection,
        content: str,
        plan: NormalizedPlan,
        corpus_chunks: Optional[list] = None,
        previous_summaries: Optional[list[str]] = None,
        factcheck_score: Optional[float] = None,
    ) -> QualityReport:
        """Évalue une section sur les 6 critères.

        Args:
            section: La section du plan.
            content: Le contenu généré.
            plan: Le plan complet.
            corpus_chunks: Les blocs de corpus utilisés.
            previous_summaries: Résumés des sections précédentes.
            factcheck_score: Score de fiabilité factuelle (0-100, du factcheck_engine).

        Returns:
            Rapport de qualité complet.
        """
        if not self.enabled:
            return QualityReport(section_id=section.id, global_score=5.0)

        report = QualityReport(section_id=section.id)
        criteria = []

        # C1, C2, C3 : évaluation par IA
        ai_scores = self._evaluate_with_ai(section, content, corpus_chunks, previous_summaries)

        # C1 — Conformité au plan
        c1_score = ai_scores.get("C1", {}).get("score", 3.0)
        criteria.append(CriterionResult(
            criterion_id="C1",
            criterion_name="Conformité au plan",
            score=float(c1_score),
            justification=ai_scores.get("C1", {}).get("justification", ""),
            weight=self.weights.get("plan_conformity", 1.0),
        ))

        # C2 — Couverture du corpus
        c2_score = ai_scores.get("C2", {}).get("score", 3.0)
        criteria.append(CriterionResult(
            criterion_id="C2",
            criterion_name="Couverture du corpus",
            score=float(c2_score),
            justification=ai_scores.get("C2", {}).get("justification", ""),
            weight=self.weights.get("corpus_coverage", 1.5),
        ))

        # C3 — Cohérence narrative
        c3_score = ai_scores.get("C3", {}).get("score", 3.0)
        criteria.append(CriterionResult(
            criterion_id="C3",
            criterion_name="Cohérence narrative",
            score=float(c3_score),
            justification=ai_scores.get("C3", {}).get("justification", ""),
            weight=self.weights.get("narrative_coherence", 0.8),
        ))

        # C4 — Respect de la taille cible (algorithmique)
        c4_result = self._evaluate_target_size(section, content)
        criteria.append(c4_result)

        # C5 — Fiabilité factuelle (depuis factcheck_engine)
        c5_result = self._evaluate_factual_reliability(factcheck_score)
        criteria.append(c5_result)

        # C6 — Traçabilité des sources (algorithmique)
        c6_result = self._evaluate_source_traceability(content)
        criteria.append(c6_result)

        report.criteria = criteria
        report.needs_source_count = len(detect_needs_source_markers(content))

        # Calcul du score global pondéré
        report.global_score = self._compute_global_score(criteria)

        # Recommandations si score < 4
        if report.global_score < 4.0:
            report.recommendations = self._generate_recommendations(criteria)

        return report

    def should_refine(self, report: QualityReport) -> bool:
        """Détermine si un raffinement est nécessaire."""
        return report.global_score < self.auto_refine_threshold

    def _evaluate_with_ai(
        self,
        section: PlanSection,
        content: str,
        corpus_chunks: Optional[list] = None,
        previous_summaries: Optional[list[str]] = None,
    ) -> dict:
        """Évalue les critères C1, C2, C3 via un appel IA."""
        if not self.provider:
            return {}

        corpus_summary = "Aucun corpus fourni."
        if corpus_chunks:
            texts = []
            for chunk in corpus_chunks[:5]:
                text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", str(chunk))
                texts.append(text[:300])
            corpus_summary = "\n---\n".join(texts)

        summaries_text = "Aucune section précédente."
        if previous_summaries:
            summaries_text = "\n".join(f"- {s}" for s in previous_summaries[-3:])

        prompt = EVALUATION_PROMPT.format(
            section_title=section.title,
            section_description=section.description or "Pas de description",
            content=content[:3000],
            corpus_summary=corpus_summary,
            previous_summaries=summaries_text,
        )

        try:
            model = self.evaluation_model or self.provider.get_default_model()
            response = self.provider.generate(
                prompt=prompt,
                system_prompt="Tu es un évaluateur de qualité. Retourne uniquement du JSON valide.",
                model=model,
                temperature=0.2,
                max_tokens=500,
            )
            return self._parse_ai_scores(response.content)
        except Exception as e:
            logger.warning(f"Évaluation IA échouée pour {section.id}: {e}")
            return {}

    @staticmethod
    def _parse_ai_scores(response_text: str) -> dict:
        """Parse la réponse JSON de l'évaluation IA."""
        from src.utils.string_utils import clean_json_string

        # Nettoyer les balises Markdown avant parsing
        text = clean_json_string(response_text)
        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                result = {}
                for key in ["C1", "C2", "C3"]:
                    if key in data:
                        score = data[key].get("score", 3)
                        score = max(1, min(5, float(score)))
                        result[key] = {
                            "score": score,
                            "justification": data[key].get("justification", ""),
                        }
                return result
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass
        return {}

    def _evaluate_target_size(self, section: PlanSection, content: str) -> CriterionResult:
        """C4 — Évalue le respect de la taille cible (algorithmique)."""
        weight = self.weights.get("target_size", 0.5)
        if not section.page_budget:
            return CriterionResult(
                criterion_id="C4",
                criterion_name="Respect de la taille cible",
                score=4.0,
                justification="Pas de budget de pages défini.",
                weight=weight,
            )

        target_words = int(section.page_budget * 400)
        actual_words = len(content.split())
        if target_words == 0:
            ratio = 1.0
        else:
            ratio = actual_words / target_words

        # Score based on deviation from target (±20% = 5, ±40% = 3, etc.)
        deviation = abs(1.0 - ratio)
        if deviation <= 0.2:
            score = 5.0
        elif deviation <= 0.35:
            score = 4.0
        elif deviation <= 0.5:
            score = 3.0
        elif deviation <= 0.7:
            score = 2.0
        else:
            score = 1.0

        return CriterionResult(
            criterion_id="C4",
            criterion_name="Respect de la taille cible",
            score=score,
            justification=f"{actual_words} mots générés vs {target_words} visés (ratio {ratio:.2f}).",
            weight=weight,
        )

    def _evaluate_factual_reliability(self, factcheck_score: Optional[float]) -> CriterionResult:
        """C5 — Score issu du factcheck_engine."""
        weight = self.weights.get("factual_reliability", 1.5)
        if factcheck_score is None:
            return CriterionResult(
                criterion_id="C5",
                criterion_name="Fiabilité factuelle",
                score=3.0,
                justification="Vérification factuelle non disponible.",
                weight=weight,
            )

        # Convert 0-100 score to 1-5 scale
        if factcheck_score >= 90:
            score = 5.0
        elif factcheck_score >= 80:
            score = 4.0
        elif factcheck_score >= 60:
            score = 3.0
        elif factcheck_score >= 40:
            score = 2.0
        else:
            score = 1.0

        return CriterionResult(
            criterion_id="C5",
            criterion_name="Fiabilité factuelle",
            score=score,
            justification=f"Score factcheck : {factcheck_score:.0f}%.",
            weight=weight,
        )

    def _evaluate_source_traceability(self, content: str) -> CriterionResult:
        """C6 — Traçabilité des sources (algorithmique)."""
        weight = self.weights.get("source_traceability", 1.2)

        # Count NEEDS_SOURCE markers (bad)
        markers = detect_needs_source_markers(content)
        marker_count = len(markers)

        # Count inline citations (good) — patterns:
        # (Author, Year), (Author et al., Year), (Author & Author, Year),
        # (Author and Author, Year), (Author, Author & Author, Year)
        citation_pattern = re.compile(
            r'\('
            r'[A-ZÀ-Ü][a-zà-ü]+'
            r'(?:\s+(?:et\s+al\.|&\s+[A-ZÀ-Ü][a-zà-ü]+|and\s+[A-ZÀ-Ü][a-zà-ü]+|,\s+[A-ZÀ-Ü][a-zà-ü]+(?:\s*&\s*[A-ZÀ-Ü][a-zà-ü]+)?))*'
            r',\s*\d{4}'
            r'\)'
        )
        citations = citation_pattern.findall(content)
        citation_count = len(citations)

        # Also count numeric bracket citations: [1], [2, 3], [1-5]
        numeric_citation_pattern = re.compile(r'\[(\d+(?:\s*[,\-–]\s*\d+)*)\]')
        numeric_citations = numeric_citation_pattern.findall(content)
        citation_count += len(numeric_citations)

        # Also count file-name citations: selon le document xxx.pdf
        file_citation_pattern = re.compile(r'(?:selon|d\'après)\s+(?:le\s+)?document\s+\S+\.(?:pdf|docx?|txt)', re.IGNORECASE)
        file_citations = file_citation_pattern.findall(content)
        citation_count += len(file_citations)

        # Score: more citations and fewer markers = better
        if marker_count == 0 and citation_count >= 3:
            score = 5.0
        elif marker_count == 0 and citation_count >= 1:
            score = 4.0
        elif marker_count <= 1 and citation_count >= 1:
            score = 3.0
        elif marker_count <= 2:
            score = 2.0
        else:
            score = 1.0

        return CriterionResult(
            criterion_id="C6",
            criterion_name="Traçabilité des sources",
            score=score,
            justification=f"{citation_count} citation(s) inline, {marker_count} marqueur(s) {{{{NEEDS_SOURCE}}}}.",
            weight=weight,
        )

    def _compute_global_score(self, criteria: list[CriterionResult]) -> float:
        """Calcule le score global pondéré."""
        total_weight = sum(c.weight for c in criteria)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(c.score * c.weight for c in criteria)
        return round(weighted_sum / total_weight, 2)

    @staticmethod
    def _generate_recommendations(criteria: list[CriterionResult]) -> list[str]:
        """Génère des recommandations basées sur les scores faibles."""
        recommendations = []
        for c in criteria:
            if c.score < 3.0:
                recommendations.append(
                    f"{c.criterion_name} (score {c.score}/5) : amélioration nécessaire — {c.justification}"
                )
            elif c.score < 4.0:
                recommendations.append(
                    f"{c.criterion_name} (score {c.score}/5) : amélioration possible."
                )
        return recommendations
