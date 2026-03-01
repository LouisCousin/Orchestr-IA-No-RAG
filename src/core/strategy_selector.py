"""Sélection dynamique de la stratégie de génération — v4.0.

Décision automatique entre trois modes :
  - STANDARD : injection directe du corpus dans chaque appel LLM
  - HIGH_VOLUME_CACHE : cache Gemini explicite (corpus ≥ 650K OU output ≥ 50K tokens)
  - SEMANTIC_COMPRESSION_REQUIRED : compression sémantique avant cache (corpus > 900K)

Deux seuils déclencheurs :
  1. Post-acquisition corpus (Input Check) : ≥ 650K tokens → HIGH_VOLUME_CACHE
  2. Post-planification (Output Check) : ≥ 50K tokens estimés → HIGH_VOLUME_CACHE
"""

import logging
from enum import Enum

logger = logging.getLogger("orchestria")


class GenerationStrategy(Enum):
    """Stratégies de génération disponibles."""
    STANDARD = "standard"
    HIGH_VOLUME_CACHE = "high_volume_cache"
    SEMANTIC_COMPRESSION_REQUIRED = "semantic_compression"


class StrategySelector:
    """Décision dynamique : standard vs full-context-cache vs compression."""

    THRESHOLD_INPUT_TOKENS: int = 650_000
    THRESHOLD_OUTPUT_TOKENS: int = 50_000
    COMPRESSION_THRESHOLD_TOKENS: int = 900_000

    def __init__(self, config: dict | None = None):
        if config:
            ctx = config.get("context_strategy", {})
            self.THRESHOLD_INPUT_TOKENS = ctx.get(
                "input_threshold_tokens", self.THRESHOLD_INPUT_TOKENS
            )
            self.THRESHOLD_OUTPUT_TOKENS = ctx.get(
                "output_threshold_tokens", self.THRESHOLD_OUTPUT_TOKENS
            )
            self.COMPRESSION_THRESHOLD_TOKENS = ctx.get(
                "compression_threshold_tokens", self.COMPRESSION_THRESHOLD_TOKENS
            )
        self._last_report: dict | None = None

    def select_strategy(
        self,
        corpus_tokens: int,
        estimated_output_tokens: int = 0,
    ) -> GenerationStrategy:
        """Sélectionne la stratégie optimale.

        Args:
            corpus_tokens: Nombre total de tokens du corpus brut.
            estimated_output_tokens: Estimation des tokens output (0 si inconnu).

        Returns:
            GenerationStrategy appropriée.
        """
        trigger = None

        if corpus_tokens >= self.COMPRESSION_THRESHOLD_TOKENS:
            strategy = GenerationStrategy.SEMANTIC_COMPRESSION_REQUIRED
            trigger = "compression_threshold"
        elif corpus_tokens >= self.THRESHOLD_INPUT_TOKENS:
            strategy = GenerationStrategy.HIGH_VOLUME_CACHE
            trigger = "input_threshold"
        elif estimated_output_tokens >= self.THRESHOLD_OUTPUT_TOKENS:
            strategy = GenerationStrategy.HIGH_VOLUME_CACHE
            trigger = "output_threshold"
        else:
            strategy = GenerationStrategy.STANDARD
            trigger = None

        self._last_report = {
            "strategy": strategy.value,
            "reason": trigger,
            "corpus_tokens": corpus_tokens,
            "estimated_output_tokens": estimated_output_tokens,
        }

        logger.info(
            f"[StrategySelector] Stratégie sélectionnée : {strategy.value} "
            f"(corpus={corpus_tokens}, output_est={estimated_output_tokens}, "
            f"trigger={trigger})"
        )
        return strategy

    def estimate_output_tokens(self, plan) -> int:
        """Estime les tokens output totaux à partir du plan.

        Formule : sum(section.target_word_count * 1.35) pour toutes les sections.
        Le facteur 1.35 compense la conversion mots → tokens en français.

        Args:
            plan: NormalizedPlan ou objet avec attribut sections.

        Returns:
            Estimation du nombre total de tokens output.
        """
        if not plan or not hasattr(plan, "sections"):
            return 0

        total = 0
        for section in plan.sections:
            word_count = getattr(section, "page_budget", None)
            if word_count is not None:
                # page_budget est en pages, ~400 mots par page
                total += int(word_count * 400 * 1.35)
            else:
                # Estimation par défaut : 500 mots par section
                total += int(500 * 1.35)

        return total

    def get_strategy_report(self) -> dict:
        """Retourne le rapport de la dernière sélection de stratégie."""
        return self._last_report or {}
