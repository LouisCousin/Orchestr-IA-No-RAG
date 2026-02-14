"""Tests unitaires pour le générateur conditionnel."""

import pytest

from src.core.conditional_generator import ConditionalGenerator, CoverageLevel, CoverageAssessment
from src.core.rag_engine import RAGResult


class TestCoverageAssessment:
    """Tests de la dataclass CoverageAssessment."""

    def test_to_dict(self):
        assessment = CoverageAssessment(
            section_id="1.1",
            section_title="Introduction",
            level=CoverageLevel.SUFFICIENT,
            avg_score=0.75,
            num_relevant_blocks=5,
            total_tokens=1000,
            message="OK",
            should_generate=True,
        )
        d = assessment.to_dict()
        assert d["section_id"] == "1.1"
        assert d["level"] == "sufficient"
        assert d["should_generate"] is True


class TestConditionalGeneratorSufficient:
    """Tests de couverture suffisante."""

    def test_sufficient_coverage(self):
        gen = ConditionalGenerator(sufficient_threshold=0.5, min_relevant_blocks=3)
        rag_result = RAGResult(
            section_id="1.1", section_title="Test",
            avg_score=0.7, num_relevant=5, total_tokens=500,
        )
        assessment = gen.assess_coverage(rag_result)
        assert assessment.level == CoverageLevel.SUFFICIENT
        assert assessment.should_generate is True
        assert assessment.extra_prompt_instruction == ""

    def test_exact_threshold(self):
        gen = ConditionalGenerator(sufficient_threshold=0.5, min_relevant_blocks=3)
        rag_result = RAGResult(
            section_id="1.1", section_title="Test",
            avg_score=0.5, num_relevant=3, total_tokens=500,
        )
        assessment = gen.assess_coverage(rag_result)
        assert assessment.level == CoverageLevel.SUFFICIENT


class TestConditionalGeneratorLow:
    """Tests de couverture faible."""

    def test_low_coverage_by_score(self):
        gen = ConditionalGenerator(
            sufficient_threshold=0.5,
            insufficient_threshold=0.3,
            min_relevant_blocks=3,
        )
        rag_result = RAGResult(
            section_id="1.2", section_title="Test",
            avg_score=0.4, num_relevant=5, total_tokens=500,
        )
        assessment = gen.assess_coverage(rag_result)
        assert assessment.level == CoverageLevel.LOW
        assert assessment.should_generate is True
        assert "ATTENTION" in assessment.extra_prompt_instruction

    def test_low_coverage_by_blocks(self):
        gen = ConditionalGenerator(
            sufficient_threshold=0.5,
            insufficient_threshold=0.3,
            min_relevant_blocks=3,
        )
        rag_result = RAGResult(
            section_id="1.2", section_title="Test",
            avg_score=0.6, num_relevant=2, total_tokens=500,
        )
        assessment = gen.assess_coverage(rag_result)
        assert assessment.level == CoverageLevel.LOW
        assert assessment.should_generate is True


class TestConditionalGeneratorInsufficient:
    """Tests de couverture insuffisante."""

    def test_insufficient_by_score(self):
        gen = ConditionalGenerator(insufficient_threshold=0.3)
        rag_result = RAGResult(
            section_id="1.3", section_title="Test",
            avg_score=0.2, num_relevant=5, total_tokens=500,
        )
        assessment = gen.assess_coverage(rag_result)
        assert assessment.level == CoverageLevel.INSUFFICIENT
        assert assessment.should_generate is False

    def test_insufficient_by_zero_blocks(self):
        gen = ConditionalGenerator(insufficient_threshold=0.3)
        rag_result = RAGResult(
            section_id="1.3", section_title="Test",
            avg_score=0.5, num_relevant=0, total_tokens=0,
        )
        assessment = gen.assess_coverage(rag_result)
        assert assessment.level == CoverageLevel.INSUFFICIENT
        assert assessment.should_generate is False

    def test_deferred_sections_tracked(self):
        gen = ConditionalGenerator(insufficient_threshold=0.3)
        rag_result = RAGResult(
            section_id="1.3", section_title="Test",
            avg_score=0.1, num_relevant=0,
        )
        gen.assess_coverage(rag_result)
        assert "1.3" in gen.deferred_sections


class TestConditionalGeneratorDisabled:
    """Tests avec génération conditionnelle désactivée."""

    def test_disabled_always_generates(self):
        gen = ConditionalGenerator(enabled=False)
        rag_result = RAGResult(
            section_id="1.1", section_title="Test",
            avg_score=0.0, num_relevant=0,
        )
        assessment = gen.assess_coverage(rag_result)
        assert assessment.should_generate is True
        assert assessment.level == CoverageLevel.SUFFICIENT


class TestConditionalGeneratorDeferredManagement:
    """Tests de gestion des sections reportées."""

    def test_clear_deferred(self):
        gen = ConditionalGenerator(insufficient_threshold=0.3)
        gen._deferred_sections = ["1.1", "1.2"]
        gen.clear_deferred()
        assert gen.deferred_sections == []

    def test_is_deferred(self):
        gen = ConditionalGenerator()
        gen._deferred_sections = ["1.1"]
        assert gen.is_deferred("1.1") is True
        assert gen.is_deferred("1.2") is False

    def test_remove_deferred(self):
        gen = ConditionalGenerator()
        gen._deferred_sections = ["1.1", "1.2"]
        gen.remove_deferred("1.1")
        assert "1.1" not in gen.deferred_sections
        assert "1.2" in gen.deferred_sections

    def test_remove_deferred_nonexistent(self):
        gen = ConditionalGenerator()
        gen._deferred_sections = ["1.1"]
        gen.remove_deferred("1.99")  # Should not raise
        assert gen.deferred_sections == ["1.1"]
