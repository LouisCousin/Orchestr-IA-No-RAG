"""Glossaire terminologique intelligent.

Phase 3 : maintient un dictionnaire de termes clés pour garantir
la cohérence terminologique dans tout le document.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.providers.base import BaseProvider
from src.utils.file_utils import save_json, load_json

logger = logging.getLogger("orchestria")

GENERATE_GLOSSARY_PROMPT = """Analyse les titres de sections et les extraits de corpus ci-dessous.
Identifie les termes techniques, acronymes, concepts clés et formulations récurrentes.

═══ TITRES DES SECTIONS ═══
{section_titles}

═══ EXTRAITS DU CORPUS ═══
{corpus_sample}

═══ INSTRUCTIONS ═══
Retourne EXACTEMENT le format JSON suivant (sans commentaires) :
{{
  "terms": [
    {{
      "term": "<terme principal>",
      "definition": "<définition courte>",
      "abbreviation": "<sigle ou null>",
      "preferred_form": "<formulation à privilégier>",
      "avoid_forms": ["<formulation à éviter>", ...],
      "domain": "<domaine thématique>"
    }},
    ...
  ]
}}
Identifie entre 5 et 20 termes pertinents."""

GENERATE_FROM_PLAN_PROMPT = """Analyse les titres et descriptions de sections ci-dessous.
Identifie les termes techniques, acronymes et concepts clés.

═══ SECTIONS DU PLAN ═══
{plan_content}

═══ INSTRUCTIONS ═══
Retourne EXACTEMENT le format JSON suivant :
{{
  "terms": [
    {{
      "term": "<terme principal>",
      "definition": "<définition courte>",
      "abbreviation": "<sigle ou null>",
      "preferred_form": "<formulation à privilégier>",
      "avoid_forms": [],
      "domain": "<domaine thématique>"
    }},
    ...
  ]
}}
Identifie entre 5 et 15 termes pertinents."""


class GlossaryEngine:
    """Gère le glossaire terminologique du projet."""

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        max_terms_per_prompt: int = 15,
        enabled: bool = False,
    ):
        self.project_dir = project_dir
        self.max_terms_per_prompt = max_terms_per_prompt
        self.enabled = enabled
        self._terms: list[dict] = []
        if project_dir:
            self._load()

    def _glossary_path(self) -> Optional[Path]:
        if self.project_dir:
            return self.project_dir / "glossary.json"
        return None

    def _load(self) -> None:
        """Charge le glossaire depuis le disque."""
        path = self._glossary_path()
        if path and path.exists():
            try:
                data = load_json(path)
                self._terms = data.get("terms", [])
            except Exception as e:
                logger.warning(f"Erreur chargement glossaire : {e}")
                self._terms = []

    def _save(self) -> None:
        """Sauvegarde le glossaire sur disque."""
        path = self._glossary_path()
        if path:
            save_json(path, {"terms": self._terms})

    # ── CRUD ──

    def add_term(
        self,
        term: str,
        definition: str,
        abbreviation: Optional[str] = None,
        preferred_form: Optional[str] = None,
        avoid_forms: Optional[list[str]] = None,
        domain: Optional[str] = None,
        source_doc_id: Optional[str] = None,
    ) -> dict:
        """Ajoute un terme au glossaire.

        Returns:
            Le terme ajouté.

        Raises:
            ValueError: Si le terme existe déjà.
        """
        if any(t["term"].lower() == term.lower() for t in self._terms):
            raise ValueError(f"Le terme '{term}' existe déjà dans le glossaire")

        entry = {
            "term": term,
            "definition": definition,
            "abbreviation": abbreviation,
            "preferred_form": preferred_form or term,
            "avoid_forms": avoid_forms or [],
            "domain": domain,
            "source_doc_id": source_doc_id,
        }
        self._terms.append(entry)
        self._save()
        return entry

    def update_term(self, term: str, **fields) -> Optional[dict]:
        """Met à jour un terme existant."""
        for t in self._terms:
            if t["term"].lower() == term.lower():
                for key, value in fields.items():
                    if key != "term":  # Don't change the key
                        t[key] = value
                self._save()
                return dict(t)
        return None

    def delete_term(self, term: str) -> bool:
        """Supprime un terme."""
        for i, t in enumerate(self._terms):
            if t["term"].lower() == term.lower():
                self._terms.pop(i)
                self._save()
                return True
        return False

    def get_term(self, term: str) -> Optional[dict]:
        """Récupère un terme par son nom."""
        for t in self._terms:
            if t["term"].lower() == term.lower():
                return dict(t)
        return None

    def get_all_terms(self) -> list[dict]:
        """Retourne tous les termes."""
        return list(self._terms)

    # ── Génération automatique ──

    def generate_from_corpus(
        self,
        plan,
        corpus_chunks: list,
        provider: BaseProvider,
        model: Optional[str] = None,
    ) -> list[dict]:
        """Génère un glossaire à partir du plan et du corpus.

        Args:
            plan: NormalizedPlan.
            corpus_chunks: Échantillon de chunks du corpus.
            provider: Fournisseur IA.
            model: Modèle à utiliser.

        Returns:
            Liste de termes proposés.
        """
        section_titles = "\n".join(
            f"- {s.title}" for s in plan.sections
        )
        corpus_sample = "\n---\n".join(
            self._get_chunk_text(chunk)[:500]
            for chunk in corpus_chunks[:10]
        )

        prompt = GENERATE_GLOSSARY_PROMPT.format(
            section_titles=section_titles,
            corpus_sample=corpus_sample,
        )

        return self._generate_terms(provider, prompt, model)

    def generate_from_plan(
        self,
        plan,
        provider: BaseProvider,
        model: Optional[str] = None,
    ) -> list[dict]:
        """Génère un glossaire à partir du plan seul (variante légère).

        Args:
            plan: NormalizedPlan.
            provider: Fournisseur IA.
            model: Modèle à utiliser.

        Returns:
            Liste de termes proposés.
        """
        plan_content = "\n".join(
            f"- {s.id} {s.title}" + (f" : {s.description}" if s.description else "")
            for s in plan.sections
        )

        prompt = GENERATE_FROM_PLAN_PROMPT.format(plan_content=plan_content)
        return self._generate_terms(provider, prompt, model)

    def _generate_terms(self, provider: BaseProvider, prompt: str, model: Optional[str] = None) -> list[dict]:
        """Appel IA commun pour la génération de termes."""
        try:
            model = model or provider.get_default_model()
            response = provider.generate(
                prompt=prompt,
                system_prompt="Tu es un terminologue expert. Retourne uniquement du JSON valide.",
                model=model,
                temperature=0.3,
                max_tokens=2000,
            )
            return self._parse_terms(response.content)
        except Exception as e:
            logger.warning(f"Génération du glossaire échouée : {e}")
            return []

    @staticmethod
    def _parse_terms(response_text: str) -> list[dict]:
        """Parse la réponse JSON de l'IA."""
        import re
        text = response_text.strip()
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                terms = data.get("terms", [])
                # Validate each term
                valid_terms = []
                for t in terms:
                    if "term" in t and "definition" in t:
                        valid_terms.append({
                            "term": t["term"],
                            "definition": t["definition"],
                            "abbreviation": t.get("abbreviation"),
                            "preferred_form": t.get("preferred_form", t["term"]),
                            "avoid_forms": t.get("avoid_forms", []),
                            "domain": t.get("domain"),
                            "source_doc_id": None,
                        })
                return valid_terms
            except (json.JSONDecodeError, ValueError):
                pass
        return []

    def apply_generated_terms(self, terms: list[dict]) -> int:
        """Applique les termes générés au glossaire (sans doublons).

        Returns:
            Nombre de termes ajoutés.
        """
        added = 0
        for t in terms:
            try:
                self.add_term(**t)
                added += 1
            except ValueError:
                pass  # Term already exists
        return added

    # ── Injection dans les prompts ──

    def get_terms_for_section(
        self,
        section_title: str,
        section_chunks: Optional[list] = None,
    ) -> list[dict]:
        """Filtre les termes pertinents pour une section.

        Sélectionne par correspondance textuelle entre le titre de section
        et les termes/domaines du glossaire. Limite à max_terms_per_prompt.

        Args:
            section_title: Titre de la section en cours.
            section_chunks: Chunks RAG de la section (optionnel).

        Returns:
            Liste des termes pertinents (max max_terms_per_prompt).
        """
        if not self._terms:
            return []

        title_lower = section_title.lower()

        # Score each term by relevance
        scored = []
        for term in self._terms:
            score = 0
            term_lower = term["term"].lower()
            domain_lower = (term.get("domain") or "").lower()

            # Direct match in title
            if term_lower in title_lower:
                score += 3
            if domain_lower and domain_lower in title_lower:
                score += 2

            # Match in chunks
            if section_chunks:
                chunk_text = " ".join(
                    self._get_chunk_text(c) for c in section_chunks[:5]
                ).lower()
                if term_lower in chunk_text:
                    score += 1

            # All terms get at least a base score
            scored.append((score, term))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:self.max_terms_per_prompt]]

    def format_for_prompt(self, terms: list[dict]) -> str:
        """Formate les termes pour injection dans un prompt.

        Returns:
            Bloc de texte formaté pour le prompt, ou chaîne vide.
        """
        if not terms:
            return ""

        lines = ["GLOSSAIRE — Utilise les termes suivants de manière cohérente :"]
        for t in terms:
            line = f"- {t['term']} : {t['definition']}."
            if t.get("preferred_form") and t["preferred_form"] != t["term"]:
                line += f" Forme préférée : {t['preferred_form']}."
            if t.get("abbreviation"):
                line += f" Abréviation : {t['abbreviation']}."
            if t.get("avoid_forms"):
                line += f" Éviter : {', '.join(t['avoid_forms'])}."
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _get_chunk_text(chunk) -> str:
        """Extrait le texte d'un chunk (dict ou objet)."""
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        return getattr(chunk, "text", str(chunk))
