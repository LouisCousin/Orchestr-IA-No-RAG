"""Tests d'intégration pour le pipeline de vérification factuelle."""

import pytest
from unittest.mock import MagicMock
from src.core.factcheck_engine import (
    FactcheckEngine,
    FactcheckReport,
    CORROBORATED,
    PLAUSIBLE,
    UNFOUNDED,
    CONTRADICTED,
)


@pytest.fixture
def provider():
    p = MagicMock()
    p.get_default_model.return_value = "gpt-4o"
    return p


@pytest.fixture
def engine(provider):
    return FactcheckEngine(
        provider=provider,
        enabled=True,
        auto_correct_threshold=80,
        max_claims_per_section=30,
    )


class TestFactcheckPipeline:
    def test_full_pipeline_corroborated(self, engine, provider):
        """Pipeline complet: extraction → évaluation individuelle (no RAG)."""
        # Without RAG, corpus_text is "", so it goes to extract + evaluate path
        extraction_response = '{"claims": [{"id": 1, "text": "Le PIB a augmenté de 3%", "type": "data"}]}'
        eval_response = '{"status": "CORROBORÉE", "justification": "Confirmed by corpus data"}'

        provider.generate.side_effect = [
            MagicMock(content=extraction_response),
            MagicMock(content=eval_response),
        ]

        report = engine.check_section(
            section_id="1.1",
            content="Le PIB a augmenté de 3% en 2024.",
            section_title="Économie",
        )

        assert isinstance(report, FactcheckReport)
        assert report.section_id == "1.1"
        assert report.reliability_score > 0
        assert len(report.details) == 1
        assert report.details[0].status == CORROBORATED

    def test_pipeline_with_contradicted_claims(self, engine, provider):
        """Pipeline avec affirmations contredites."""
        extraction_response = '{"claims": [{"id": 1, "text": "Revenue was 10M", "type": "data"}, {"id": 2, "text": "Founded in 2020", "type": "fact"}]}'
        eval_response_1 = '{"status": "CONTREDITE", "justification": "Corpus says 5M"}'
        eval_response_2 = '{"status": "CORROBORÉE", "justification": "Confirmed"}'

        provider.generate.side_effect = [
            MagicMock(content=extraction_response),
            MagicMock(content=eval_response_1),
            MagicMock(content=eval_response_2),
        ]

        report = engine.check_section(
            section_id="2.1",
            content="Revenue was 10M. Founded in 2020.",
            section_title="Company info",
        )

        assert report.reliability_score < 100

    def test_pipeline_disabled(self, provider):
        """Pipeline désactivé retourne score 100."""
        engine = FactcheckEngine(provider=provider, enabled=False)
        result = engine.check_section(
            section_id="1.1",
            content="Any content",
        )
        assert result.reliability_score == 100.0

    def test_pipeline_no_provider(self):
        """Sans provider, retourne score 100."""
        engine = FactcheckEngine(provider=None, enabled=True)
        result = engine.check_section(
            section_id="1.1",
            content="Content",
        )
        assert result.reliability_score == 100.0

    def test_should_correct_threshold(self, engine, provider):
        """should_correct reflète le seuil auto_correct_threshold."""
        extraction_response = '{"claims": [{"id": 1, "text": "False claim", "type": "fact"}]}'
        eval_response = '{"status": "CONTREDITE", "justification": "Wrong"}'

        provider.generate.side_effect = [
            MagicMock(content=extraction_response),
            MagicMock(content=eval_response),
        ]

        report = engine.check_section(
            section_id="1.1",
            content="False claim.",
            section_title="Test",
        )

        # Reliability score: 0% (1 contradicted / 1 total = 0 reliable)
        assert engine.should_correct(report) is True

    def test_pipeline_no_claims_extracted(self, engine, provider):
        """Aucune affirmation extraite donne un score à 100."""
        provider.generate.return_value = MagicMock(content='{"claims": []}')

        report = engine.check_section(
            section_id="1.1",
            content="Content with no verifiable claims.",
            section_title="Test",
        )

        assert report.reliability_score == 100.0
        assert len(report.details) == 0

    def test_report_to_dict(self, engine, provider):
        """Sérialisation du rapport."""
        extraction_response = '{"claims": [{"id": 1, "text": "Test", "type": "fact"}]}'
        eval_response = '{"status": "PLAUSIBLE", "justification": "Likely"}'

        provider.generate.side_effect = [
            MagicMock(content=extraction_response),
            MagicMock(content=eval_response),
        ]

        report = engine.check_section(
            section_id="1.1",
            content="Test.",
            section_title="Test",
        )

        d = report.to_dict()
        assert "section_id" in d
        assert "reliability_score" in d
        assert "details" in d
