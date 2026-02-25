"""Tests unitaires pour quality_evaluator.py."""

import pytest
from unittest.mock import MagicMock, patch
from src.core.quality_evaluator import QualityEvaluator, QualityReport, CriterionResult
from src.core.plan_parser import PlanSection, NormalizedPlan


@pytest.fixture
def section():
    return PlanSection(id="1.1", title="Introduction", level=1, description="Section d'introduction", page_budget=2.0)


@pytest.fixture
def plan():
    p = NormalizedPlan(title="Test Plan")
    p.sections = [
        PlanSection(id="1.1", title="Introduction", level=1, page_budget=2.0),
        PlanSection(id="1.2", title="Analyse", level=1, page_budget=3.0),
    ]
    p.objective = "Document de test"
    return p


@pytest.fixture
def evaluator():
    return QualityEvaluator(provider=None, enabled=True)


class TestQualityEvaluator:
    def test_disabled_returns_perfect_score(self, section, plan):
        ev = QualityEvaluator(enabled=False)
        report = ev.evaluate_section(section, "Some content", plan)
        assert report.global_score == 5.0

    def test_evaluate_target_size_within_range(self, evaluator, section, plan):
        # 2 pages * 400 = 800 words target, generate ~800 words
        content = " ".join(["word"] * 800)
        result = evaluator._evaluate_target_size(section, content)
        assert result.score == 5.0
        assert result.criterion_id == "C4"

    def test_evaluate_target_size_too_short(self, evaluator, section, plan):
        content = " ".join(["word"] * 200)  # ~25% of target
        result = evaluator._evaluate_target_size(section, content)
        assert result.score < 3.0

    def test_evaluate_target_size_no_budget(self, evaluator):
        section = PlanSection(id="1.1", title="Test", level=1)
        result = evaluator._evaluate_target_size(section, "content")
        assert result.score == 4.0  # Default when no budget

    def test_evaluate_factual_reliability_high(self, evaluator):
        result = evaluator._evaluate_factual_reliability(95.0)
        assert result.score == 5.0

    def test_evaluate_factual_reliability_low(self, evaluator):
        result = evaluator._evaluate_factual_reliability(30.0)
        assert result.score == 1.0

    def test_evaluate_factual_reliability_none(self, evaluator):
        result = evaluator._evaluate_factual_reliability(None)
        assert result.score == 3.0

    def test_evaluate_source_traceability_good(self, evaluator):
        content = "Selon (Dupont, 2024) et (Smith, 2023), l'analyse montre que (Martin, 2022) a confirmé."
        result = evaluator._evaluate_source_traceability(content)
        assert result.score >= 4.0

    def test_evaluate_source_traceability_with_markers(self, evaluator):
        content = "Contenu avec {{NEEDS_SOURCE: point A}} et {{NEEDS_SOURCE: point B}} et {{NEEDS_SOURCE: point C}}."
        result = evaluator._evaluate_source_traceability(content)
        assert result.score <= 2.0

    def test_compute_global_score(self, evaluator):
        criteria = [
            CriterionResult("C1", "Test", 5.0, weight=1.0),
            CriterionResult("C2", "Test", 3.0, weight=2.0),
        ]
        score = evaluator._compute_global_score(criteria)
        # (5*1 + 3*2) / (1+2) = 11/3 = 3.67
        assert abs(score - 3.67) < 0.1

    def test_should_refine(self, evaluator):
        report = QualityReport(section_id="1.1", global_score=2.5)
        assert evaluator.should_refine(report) is True
        report.global_score = 4.0
        assert evaluator.should_refine(report) is False

    def test_generate_recommendations(self):
        criteria = [
            CriterionResult("C1", "Conformité", 2.0),
            CriterionResult("C2", "Couverture", 4.5),
        ]
        recs = QualityEvaluator._generate_recommendations(criteria)
        assert len(recs) == 1
        assert "Conformité" in recs[0]

    def test_report_serialization(self):
        report = QualityReport(
            section_id="1.1",
            global_score=3.5,
            needs_source_count=2,
            recommendations=["Fix X"],
        )
        report.criteria.append(CriterionResult("C1", "Test", 4.0, "Good", 1.0))
        d = report.to_dict()
        assert d["section_id"] == "1.1"
        assert d["global_score"] == 3.5
        assert len(d["criteria"]) == 1

        # Round-trip
        report2 = QualityReport.from_dict(d)
        assert report2.section_id == "1.1"
        assert report2.global_score == 3.5

    def test_parse_ai_scores_valid(self):
        text = '{"C1": {"score": 4, "justification": "Good"}, "C2": {"score": 3, "justification": "OK"}, "C3": {"score": 5, "justification": "Great"}}'
        result = QualityEvaluator._parse_ai_scores(text)
        assert result["C1"]["score"] == 4
        assert result["C3"]["score"] == 5

    def test_parse_ai_scores_invalid(self):
        result = QualityEvaluator._parse_ai_scores("not json at all")
        assert result == {}

    def test_parse_ai_scores_clamps_values(self):
        text = '{"C1": {"score": 10, "justification": "Over"}, "C2": {"score": 0, "justification": "Under"}}'
        result = QualityEvaluator._parse_ai_scores(text)
        assert result["C1"]["score"] == 5  # Clamped to max
        assert result["C2"]["score"] == 1  # Clamped to min
