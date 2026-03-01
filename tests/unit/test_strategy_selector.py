"""Tests unitaires pour le StrategySelector — v4.0."""

import pytest

from src.core.strategy_selector import GenerationStrategy, StrategySelector


class TestStrategySelector:
    """Tests de la sélection de stratégie."""

    def setup_method(self):
        self.selector = StrategySelector()

    def test_strategy_input_below_threshold(self):
        """Corpus < 650K tokens et pas d'estimation output → STANDARD."""
        strategy = self.selector.select_strategy(corpus_tokens=100_000)
        assert strategy == GenerationStrategy.STANDARD

    def test_strategy_input_above_threshold(self):
        """Corpus ≥ 650K tokens → HIGH_VOLUME_CACHE."""
        strategy = self.selector.select_strategy(corpus_tokens=700_000)
        assert strategy == GenerationStrategy.HIGH_VOLUME_CACHE

    def test_strategy_input_at_exact_threshold(self):
        """Corpus == 650K tokens → HIGH_VOLUME_CACHE."""
        strategy = self.selector.select_strategy(corpus_tokens=650_000)
        assert strategy == GenerationStrategy.HIGH_VOLUME_CACHE

    def test_strategy_output_above_threshold(self):
        """Output estimé ≥ 50K → HIGH_VOLUME_CACHE même si corpus petit."""
        strategy = self.selector.select_strategy(
            corpus_tokens=100_000,
            estimated_output_tokens=60_000,
        )
        assert strategy == GenerationStrategy.HIGH_VOLUME_CACHE

    def test_strategy_both_thresholds(self):
        """Les deux seuils dépassés → HIGH_VOLUME_CACHE."""
        strategy = self.selector.select_strategy(
            corpus_tokens=700_000,
            estimated_output_tokens=60_000,
        )
        assert strategy == GenerationStrategy.HIGH_VOLUME_CACHE

    def test_strategy_compression_required(self):
        """Corpus > 900K tokens → SEMANTIC_COMPRESSION_REQUIRED."""
        strategy = self.selector.select_strategy(corpus_tokens=950_000)
        assert strategy == GenerationStrategy.SEMANTIC_COMPRESSION_REQUIRED

    def test_strategy_report(self):
        """get_strategy_report retourne le bon rapport."""
        self.selector.select_strategy(
            corpus_tokens=700_000,
            estimated_output_tokens=30_000,
        )
        report = self.selector.get_strategy_report()
        assert report["strategy"] == "high_volume_cache"
        assert report["reason"] == "input_threshold"
        assert report["corpus_tokens"] == 700_000
        assert report["estimated_output_tokens"] == 30_000

    def test_strategy_report_empty_before_selection(self):
        """get_strategy_report retourne {} avant toute sélection."""
        report = self.selector.get_strategy_report()
        assert report == {}

    def test_custom_thresholds_from_config(self):
        """Les seuils personnalisés de la config sont respectés."""
        config = {
            "context_strategy": {
                "input_threshold_tokens": 500_000,
                "output_threshold_tokens": 30_000,
                "compression_threshold_tokens": 800_000,
            }
        }
        selector = StrategySelector(config=config)

        strategy = selector.select_strategy(corpus_tokens=550_000)
        assert strategy == GenerationStrategy.HIGH_VOLUME_CACHE

        strategy = selector.select_strategy(corpus_tokens=450_000)
        assert strategy == GenerationStrategy.STANDARD

    def test_estimate_output_tokens_with_plan(self):
        """estimate_output_tokens calcule correctement à partir du plan."""

        class MockSection:
            def __init__(self, page_budget):
                self.page_budget = page_budget

        class MockPlan:
            def __init__(self, sections):
                self.sections = sections

        plan = MockPlan([
            MockSection(page_budget=5),
            MockSection(page_budget=10),
            MockSection(page_budget=3),
        ])

        result = self.selector.estimate_output_tokens(plan)
        expected = int((5 + 10 + 3) * 400 * 1.35)
        assert result == expected

    def test_estimate_output_tokens_none_plan(self):
        """estimate_output_tokens retourne 0 pour un plan None."""
        assert self.selector.estimate_output_tokens(None) == 0


    def test_strategy_atlas_threshold(self):
        """Corpus > 3.2M tokens → CORPUS_ATLAS."""
        strategy = self.selector.select_strategy(corpus_tokens=4_000_000)
        assert strategy == GenerationStrategy.CORPUS_ATLAS

    def test_strategy_atlas_at_threshold(self):
        """Corpus == 3.2M tokens → CORPUS_ATLAS."""
        strategy = self.selector.select_strategy(corpus_tokens=3_200_000)
        assert strategy == GenerationStrategy.CORPUS_ATLAS

    def test_strategy_atlas_just_below(self):
        """Corpus just below 3.2M → SEMANTIC_COMPRESSION_REQUIRED."""
        strategy = self.selector.select_strategy(corpus_tokens=3_199_999)
        assert strategy == GenerationStrategy.SEMANTIC_COMPRESSION_REQUIRED

    def test_strategy_atlas_report(self):
        """Atlas trigger in report."""
        self.selector.select_strategy(corpus_tokens=5_000_000)
        report = self.selector.get_strategy_report()
        assert report["strategy"] == "corpus_atlas"
        assert report["reason"] == "atlas_threshold"

    def test_custom_atlas_threshold_from_config(self):
        """Custom atlas threshold from config."""
        config = {
            "context_strategy": {
                "atlas_threshold_tokens": 2_000_000,
            }
        }
        selector = StrategySelector(config=config)
        strategy = selector.select_strategy(corpus_tokens=2_500_000)
        assert strategy == GenerationStrategy.CORPUS_ATLAS


class TestGenerationStrategy:
    """Tests de l'enum GenerationStrategy."""

    def test_standard_value(self):
        assert GenerationStrategy.STANDARD.value == "standard"

    def test_high_volume_cache_value(self):
        assert GenerationStrategy.HIGH_VOLUME_CACHE.value == "high_volume_cache"

    def test_compression_value(self):
        assert GenerationStrategy.SEMANTIC_COMPRESSION_REQUIRED.value == "semantic_compression"

    def test_atlas_value(self):
        assert GenerationStrategy.CORPUS_ATLAS.value == "corpus_atlas"
