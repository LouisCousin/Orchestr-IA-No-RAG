"""Tests unitaires pour le module cost_tracker."""

import pytest

from src.core.cost_tracker import CostTracker, CostEntry, CostReport


@pytest.fixture
def tracker():
    return CostTracker()


class TestCalculateCost:
    def test_gpt4o_cost(self, tracker):
        cost = tracker.calculate_cost("openai", "gpt-4o", 1000, 500)
        # gpt-4o: $2.50/M input, $10.00/M output
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001

    def test_gpt4o_mini_cost(self, tracker):
        cost = tracker.calculate_cost("openai", "gpt-4o-mini", 10000, 5000)
        # gpt-4o-mini: $0.15/M input, $0.60/M output
        expected = (10000 / 1_000_000) * 0.15 + (5000 / 1_000_000) * 0.60
        assert abs(cost - expected) < 0.0001

    def test_unknown_model_returns_zero(self, tracker):
        cost = tracker.calculate_cost("unknown", "unknown-model", 1000, 500)
        assert cost == 0.0

    def test_zero_tokens(self, tracker):
        cost = tracker.calculate_cost("openai", "gpt-4o", 0, 0)
        assert cost == 0.0


class TestRecord:
    def test_record_entry(self, tracker):
        entry = tracker.record(
            section_id="1",
            model="gpt-4o",
            provider="openai",
            input_tokens=1000,
            output_tokens=500,
            task_type="generation",
        )
        assert entry.cost_usd > 0
        assert entry.section_id == "1"
        assert tracker.report.total_input_tokens == 1000
        assert tracker.report.total_output_tokens == 500

    def test_cumulative_recording(self, tracker):
        tracker.record("1", "gpt-4o", "openai", 1000, 500)
        tracker.record("2", "gpt-4o", "openai", 2000, 1000)
        assert tracker.report.total_input_tokens == 3000
        assert tracker.report.total_output_tokens == 1500
        assert len(tracker.report.entries) == 2


class TestEstimateProjectCost:
    def test_basic_estimate(self, tracker):
        estimate = tracker.estimate_project_cost(
            section_count=5,
            avg_corpus_tokens=2000,
            provider="openai",
            model="gpt-4o",
        )
        assert "error" not in estimate
        assert estimate["estimated_cost_usd"] > 0
        assert estimate["section_count"] == 5
        assert estimate["context_window"] == 128000

    def test_multi_pass_increases_cost(self, tracker):
        est1 = tracker.estimate_project_cost(5, 2000, "openai", "gpt-4o", num_passes=1)
        est2 = tracker.estimate_project_cost(5, 2000, "openai", "gpt-4o", num_passes=2)
        assert est2["estimated_cost_usd"] > est1["estimated_cost_usd"]

    def test_unknown_model(self, tracker):
        estimate = tracker.estimate_project_cost(5, 2000, "unknown", "unknown")
        assert "error" in estimate


class TestEstimateMultiModel:
    def test_returns_sorted(self, tracker):
        estimates = tracker.estimate_multi_model(5, 2000)
        assert len(estimates) > 0
        # Vérifie le tri par coût croissant
        for i in range(len(estimates) - 1):
            assert estimates[i]["estimated_cost_usd"] <= estimates[i + 1]["estimated_cost_usd"]


class TestEstimateCorpusCost:
    def test_corpus_cost(self, tracker):
        documents = [
            {"tokens": 5000},
            {"tokens": 10000},
            {"tokens": 3000},
        ]
        estimate = tracker.estimate_corpus_cost(documents, "openai", "gpt-4o")
        assert "error" not in estimate
        assert estimate["total_tokens"] == 18000
        assert estimate["estimated_input_cost_usd"] > 0

    def test_context_window_warning(self, tracker):
        documents = [
            {"tokens": 200000},  # Dépasse la fenêtre de gpt-4o (128000)
        ]
        estimate = tracker.estimate_corpus_cost(documents, "openai", "gpt-4o")
        assert estimate["documents_exceeding_context"] == 1


class TestCostReport:
    def test_to_dict(self, tracker):
        tracker.record("1", "gpt-4o", "openai", 1000, 500)
        data = tracker.report.to_dict()
        assert "entries" in data
        assert "total_cost_usd" in data
        assert len(data["entries"]) == 1

    def test_reset(self, tracker):
        tracker.record("1", "gpt-4o", "openai", 1000, 500)
        tracker.reset()
        assert tracker.report.total_input_tokens == 0
        assert len(tracker.report.entries) == 0
