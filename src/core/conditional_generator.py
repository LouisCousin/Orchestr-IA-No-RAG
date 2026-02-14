"""Génération conditionnelle de sections basée sur le score RAG (Phase 2).

Évalue la couverture du corpus avant la génération de chaque section
et décide si la génération doit procéder, être avertie ou reportée.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.core.rag_engine import RAGResult

logger = logging.getLogger("orchestria")


class CoverageLevel(str, Enum):
    """Niveau de couverture du corpus pour une section."""
    SUFFICIENT = "sufficient"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class CoverageAssessment:
    """Résultat de l'évaluation de la couverture du corpus pour une section."""
    section_id: str
    section_title: str
    level: CoverageLevel
    avg_score: float
    num_relevant_blocks: int
    total_tokens: int
    message: str
    should_generate: bool
    extra_prompt_instruction: str = ""

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "level": self.level.value,
            "avg_score": round(self.avg_score, 4),
            "num_relevant_blocks": self.num_relevant_blocks,
            "total_tokens": self.total_tokens,
            "message": self.message,
            "should_generate": self.should_generate,
        }


class ConditionalGenerator:
    """Évalue la couverture du corpus et décide de la génération.

    Trois niveaux de couverture :
    - Suffisante : score moyen >= sufficient_threshold ET >= min_relevant_blocks blocs
    - Faible : entre les seuils suffisant et insuffisant
    - Insuffisante : score moyen < insufficient_threshold OU 0 bloc pertinent
    """

    def __init__(
        self,
        sufficient_threshold: float = 0.5,
        insufficient_threshold: float = 0.3,
        min_relevant_blocks: int = 3,
        enabled: bool = True,
    ):
        self.sufficient_threshold = sufficient_threshold
        self.insufficient_threshold = insufficient_threshold
        self.min_relevant_blocks = min_relevant_blocks
        self.enabled = enabled
        self._deferred_sections: list[str] = []

    def assess_coverage(self, rag_result: RAGResult) -> CoverageAssessment:
        """Évalue la couverture du corpus pour une section.

        Args:
            rag_result: Résultat de la recherche RAG pour la section.

        Returns:
            CoverageAssessment avec le niveau et la décision.
        """
        if not self.enabled:
            return CoverageAssessment(
                section_id=rag_result.section_id,
                section_title=rag_result.section_title,
                level=CoverageLevel.SUFFICIENT,
                avg_score=rag_result.avg_score,
                num_relevant_blocks=rag_result.num_relevant,
                total_tokens=rag_result.total_tokens,
                message="Génération conditionnelle désactivée",
                should_generate=True,
            )

        avg_score = rag_result.avg_score
        num_relevant = rag_result.num_relevant

        # Couverture suffisante
        if avg_score >= self.sufficient_threshold and num_relevant >= self.min_relevant_blocks:
            return CoverageAssessment(
                section_id=rag_result.section_id,
                section_title=rag_result.section_title,
                level=CoverageLevel.SUFFICIENT,
                avg_score=avg_score,
                num_relevant_blocks=num_relevant,
                total_tokens=rag_result.total_tokens,
                message=f"Couverture suffisante ({num_relevant} blocs pertinents, score moyen {avg_score:.2f})",
                should_generate=True,
            )

        # Couverture insuffisante
        if avg_score < self.insufficient_threshold or num_relevant == 0:
            self._deferred_sections.append(rag_result.section_id)
            return CoverageAssessment(
                section_id=rag_result.section_id,
                section_title=rag_result.section_title,
                level=CoverageLevel.INSUFFICIENT,
                avg_score=avg_score,
                num_relevant_blocks=num_relevant,
                total_tokens=rag_result.total_tokens,
                message=f"Couverture insuffisante ({num_relevant} blocs, score moyen {avg_score:.2f}). "
                        f"Section reportée.",
                should_generate=False,
            )

        # Couverture faible
        return CoverageAssessment(
            section_id=rag_result.section_id,
            section_title=rag_result.section_title,
            level=CoverageLevel.LOW,
            avg_score=avg_score,
            num_relevant_blocks=num_relevant,
            total_tokens=rag_result.total_tokens,
            message=f"Couverture faible ({num_relevant} blocs, score moyen {avg_score:.2f}). "
                    f"Génération avec avertissement.",
            should_generate=True,
            extra_prompt_instruction=(
                "ATTENTION : Le corpus source contient peu d'informations directement liées "
                "à cette section. Signale clairement les passages pour lesquels tu manques "
                "de sources documentaires. Ne fabrique pas de données."
            ),
        )

    @property
    def deferred_sections(self) -> list[str]:
        """Retourne la liste des sections reportées."""
        return self._deferred_sections.copy()

    def clear_deferred(self) -> None:
        """Réinitialise la liste des sections reportées."""
        self._deferred_sections.clear()

    def is_deferred(self, section_id: str) -> bool:
        """Vérifie si une section est reportée."""
        return section_id in self._deferred_sections

    def remove_deferred(self, section_id: str) -> None:
        """Retire une section de la liste des reportées (après réévaluation)."""
        if section_id in self._deferred_sections:
            self._deferred_sections.remove(section_id)
