"""Feedback loop — Apprentissage par correction humaine.

Phase 3 : analyse les écarts entre le contenu généré et les corrections
humaines pour proposer des améliorations de prompts.
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional

from src.providers.base import BaseProvider

logger = logging.getLogger("orchestria")

ANALYSIS_PROMPT = """Analyse les différences entre le texte original et le texte corrigé ci-dessous.

═══ TEXTE ORIGINAL (généré par IA) ═══
{original_text}

═══ TEXTE CORRIGÉ (par l'utilisateur) ═══
{corrected_text}

═══ INSTRUCTIONS ═══
1. Identifie la catégorie dominante de correction :
   - style : corrections de ton, registre, vocabulaire
   - structure : réorganisation, ajouts/suppressions de paragraphes
   - contenu : corrections de faits, données, références
   - longueur : texte raccourci ou allongé

2. Propose une modification du prompt de génération pour éviter cette correction à l'avenir.

Retourne EXACTEMENT le format JSON suivant :
{{
  "category": "style|structure|contenu|longueur",
  "analysis": "<description des modifications en 2-3 phrases>",
  "prompt_suggestion": "<consigne à ajouter au prompt pour les sections suivantes>"
}}"""


class FeedbackEntry:
    """Entrée du journal de feedback."""

    def __init__(
        self,
        section_id: str,
        category: str,
        suggestion: str,
        decision: str = "pending",
        original_prompt: str = "",
        modified_prompt: str = "",
        analysis: str = "",
    ):
        self.section_id = section_id
        self.category = category
        self.suggestion = suggestion
        self.decision = decision  # pending, accepted, rejected, modified
        self.original_prompt = original_prompt
        self.modified_prompt = modified_prompt
        self.analysis = analysis
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "category": self.category,
            "suggestion": self.suggestion,
            "decision": self.decision,
            "original_prompt": self.original_prompt[:500],
            "modified_prompt": self.modified_prompt[:500],
            "analysis": self.analysis,
            "timestamp": self.timestamp,
        }


class FeedbackEngine:
    """Analyse les corrections humaines et propose des améliorations."""

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        enabled: bool = True,
        min_diff_ratio: float = 0.15,
        analysis_model: Optional[str] = None,
    ):
        self.provider = provider
        self.enabled = enabled
        self.min_diff_ratio = min_diff_ratio
        self.analysis_model = analysis_model
        self._history: list[FeedbackEntry] = []
        self._pending_suggestions: list[FeedbackEntry] = []

    def detect_modification(self, original: str, corrected: str) -> bool:
        """Détecte si la modification dépasse le seuil.

        Args:
            original: Texte original.
            corrected: Texte corrigé.

        Returns:
            True si la modification est significative.
        """
        if not original or not corrected:
            return False
        ratio = self._levenshtein_ratio(original, corrected)
        return ratio >= self.min_diff_ratio

    def analyze_modification(
        self,
        section_id: str,
        original: str,
        corrected: str,
    ) -> Optional[FeedbackEntry]:
        """Analyse une modification et génère une proposition.

        Args:
            section_id: Identifiant de la section.
            original: Texte original.
            corrected: Texte corrigé.

        Returns:
            FeedbackEntry avec la proposition, ou None si pas d'analyse possible.
        """
        if not self.enabled or not self.provider:
            return None

        if not self.detect_modification(original, corrected):
            return None

        prompt = ANALYSIS_PROMPT.format(
            original_text=original[:3000],
            corrected_text=corrected[:3000],
        )

        try:
            model = self.analysis_model or self.provider.get_default_model()
            response = self.provider.generate(
                prompt=prompt,
                system_prompt="Tu es un analyste de qualité rédactionnelle. Retourne uniquement du JSON valide.",
                model=model,
                temperature=0.3,
                max_tokens=500,
            )
            data = self._parse_json(response.content)
            if not data:
                return None

            entry = FeedbackEntry(
                section_id=section_id,
                category=data.get("category", "style"),
                suggestion=data.get("prompt_suggestion", ""),
                analysis=data.get("analysis", ""),
            )
            self._pending_suggestions.append(entry)
            return entry

        except Exception as e:
            logger.warning(f"Analyse feedback échouée pour {section_id}: {e}")
            return None

    def accept_suggestion(self, entry: FeedbackEntry) -> None:
        """Accepte une proposition de modification."""
        entry.decision = "accepted"
        self._history.append(entry)
        self._remove_pending(entry)

    def reject_suggestion(self, entry: FeedbackEntry) -> None:
        """Rejette une proposition."""
        entry.decision = "rejected"
        self._history.append(entry)
        self._remove_pending(entry)

    def modify_suggestion(self, entry: FeedbackEntry, modified_text: str) -> None:
        """Accepte une proposition avec modification."""
        entry.decision = "modified"
        entry.modified_prompt = modified_text
        self._history.append(entry)
        self._remove_pending(entry)

    def _remove_pending(self, entry: FeedbackEntry) -> None:
        """Retire une entrée des suggestions en attente."""
        self._pending_suggestions = [
            s for s in self._pending_suggestions
            if s.section_id != entry.section_id or s.timestamp != entry.timestamp
        ]

    @property
    def pending_suggestions(self) -> list[FeedbackEntry]:
        """Retourne les suggestions en attente."""
        return list(self._pending_suggestions)

    @property
    def history(self) -> list[dict]:
        """Retourne l'historique des feedback."""
        return [e.to_dict() for e in self._history]

    def get_statistics(self) -> dict:
        """Retourne les statistiques du feedback loop."""
        total = len(self._history)
        if total == 0:
            return {
                "total": 0,
                "acceptance_rate": 0.0,
                "categories": {},
            }

        accepted = sum(1 for e in self._history if e.decision == "accepted")
        modified = sum(1 for e in self._history if e.decision == "modified")
        rejected = sum(1 for e in self._history if e.decision == "rejected")

        categories: dict[str, int] = {}
        for e in self._history:
            cat = e.category
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total": total,
            "accepted": accepted,
            "modified": modified,
            "rejected": rejected,
            "acceptance_rate": (accepted + modified) / total if total > 0 else 0.0,
            "categories": categories,
        }

    def get_active_adjustments(self) -> list[str]:
        """Retourne les ajustements actifs (suggestions acceptées ou modifiées).

        Utilisé pour enrichir les prompts des sections suivantes.
        """
        adjustments = []
        for entry in self._history:
            if entry.decision == "accepted" and entry.suggestion:
                adjustments.append(entry.suggestion)
            elif entry.decision == "modified" and entry.modified_prompt:
                adjustments.append(entry.modified_prompt)
        return adjustments

    @staticmethod
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        """Calcule le ratio de Levenshtein normalisé.

        Retourne 0.0 (identiques) à 1.0 (totalement différents).
        """
        try:
            import Levenshtein
            return 1.0 - Levenshtein.ratio(s1, s2)
        except ImportError:
            # Fallback: simple word-level diff ratio
            words1 = set(s1.lower().split())
            words2 = set(s2.lower().split())
            if not words1 and not words2:
                return 0.0
            union = words1 | words2
            intersection = words1 & words2
            if not union:
                return 0.0
            return 1.0 - (len(intersection) / len(union))

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse une réponse JSON."""
        text = text.strip()
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}
