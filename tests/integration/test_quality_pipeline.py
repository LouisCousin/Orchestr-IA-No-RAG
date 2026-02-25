"""Tests d'intégration pour le pipeline de qualité (quality_evaluator + factcheck)."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional
from src.core.quality_evaluator import QualityEvaluator, QualityReport, DEFAULT_WEIGHTS


@dataclass
class MockSection:
    id: str = "1.1"
    title: str = "Introduction"
    description: str = "Section d'introduction"
    page_budget: Optional[float] = 2.0


@dataclass
class MockPlan:
    sections: list = None
    target_pages: int = 10

    def __post_init__(self):
        if self.sections is None:
            self.sections = [MockSection()]


@pytest.fixture
def provider():
    p = MagicMock()
    p.get_default_model.return_value = "gpt-4o"
    return p


@pytest.fixture
def evaluator(provider):
    return QualityEvaluator(
        provider=provider,
        enabled=True,
        auto_refine_threshold=3.0,
        weights=DEFAULT_WEIGHTS,
    )


class TestQualityPipeline:
    def test_full_evaluation_flow(self, evaluator, provider):
        """Test complet : évaluation AI + critères algorithmiques."""
        ai_response = '{"C1": {"score": 4, "justification": "Good"}, "C2": {"score": 3, "justification": "OK"}, "C3": {"score": 4, "justification": "Consistent"}}'
        provider.generate.return_value = MagicMock(content=ai_response)

        report = evaluator.evaluate_section(
            section=MockSection(),
            content="Test content with (Dupont, 2024) reference. " * 20,
            plan=MockPlan(),
            corpus_chunks=[{"text": "Source data"}],
            previous_summaries=["Previous section summary"],
            factcheck_score=85.0,
        )

        assert isinstance(report, QualityReport)
        assert report.section_id == "1.1"
        assert len(report.criteria) == 6
        assert report.global_score > 0

    def test_evaluation_with_low_factcheck(self, evaluator, provider):
        """Le score factcheck bas impacte le score global."""
        ai_response = '{"C1": {"score": 5, "justification": "Perfect"}, "C2": {"score": 5, "justification": "Perfect"}, "C3": {"score": 5, "justification": "Perfect"}}'
        provider.generate.return_value = MagicMock(content=ai_response)

        report_good = evaluator.evaluate_section(
            section=MockSection(),
            content="Content (Dupont, 2024)." * 10,
            plan=MockPlan(),
            factcheck_score=95.0,
        )

        report_bad = evaluator.evaluate_section(
            section=MockSection(),
            content="Content (Dupont, 2024)." * 10,
            plan=MockPlan(),
            factcheck_score=30.0,
        )

        assert report_good.global_score > report_bad.global_score

    def test_evaluation_disabled(self, provider):
        """Évaluation désactivée retourne un rapport avec score 5.0."""
        evaluator = QualityEvaluator(provider=provider, enabled=False)
        result = evaluator.evaluate_section(
            section=MockSection(),
            content="Content",
            plan=MockPlan(),
        )
        assert isinstance(result, QualityReport)
        assert result.global_score == 5.0

    def test_evaluation_recommendations(self, evaluator, provider):
        """Les recommandations sont générées pour les scores bas."""
        ai_response = '{"C1": {"score": 2, "justification": "Off topic"}, "C2": {"score": 1, "justification": "No corpus"}, "C3": {"score": 2, "justification": "Inconsistent"}}'
        provider.generate.return_value = MagicMock(content=ai_response)

        report = evaluator.evaluate_section(
            section=MockSection(),
            content="Short content",
            plan=MockPlan(),
            factcheck_score=40.0,
        )

        assert len(report.recommendations) > 0

    def test_needs_refinement(self, evaluator, provider):
        """should_refine détecte les sections à reprendre."""
        ai_response = '{"C1": {"score": 1, "justification": "Bad"}, "C2": {"score": 1, "justification": "Bad"}, "C3": {"score": 1, "justification": "Bad"}}'
        provider.generate.return_value = MagicMock(content=ai_response)

        report = evaluator.evaluate_section(
            section=MockSection(),
            content="Bad content",
            plan=MockPlan(),
            factcheck_score=10.0,
        )

        assert evaluator.should_refine(report) is True

    def test_report_serialization(self, evaluator, provider):
        """Le rapport est sérialisable en dict."""
        ai_response = '{"C1": {"score": 4, "justification": "OK"}, "C2": {"score": 3, "justification": "OK"}, "C3": {"score": 4, "justification": "OK"}}'
        provider.generate.return_value = MagicMock(content=ai_response)

        report = evaluator.evaluate_section(
            section=MockSection(),
            content="Test content (Dupont, 2024)",
            plan=MockPlan(),
            factcheck_score=80.0,
        )

        d = report.to_dict()
        assert "section_id" in d
        assert "global_score" in d
        assert "criteria" in d
        assert "recommendations" in d
