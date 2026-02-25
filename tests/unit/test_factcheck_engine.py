"""Tests unitaires pour factcheck_engine.py."""

import pytest
from unittest.mock import MagicMock
from src.core.factcheck_engine import (
    FactcheckEngine, FactcheckReport, ClaimResult,
    CORROBORATED, PLAUSIBLE, UNFOUNDED, CONTRADICTED,
)


@pytest.fixture
def engine():
    return FactcheckEngine(provider=None, enabled=True)


class TestFactcheckEngine:
    def test_disabled_returns_perfect_score(self):
        engine = FactcheckEngine(enabled=False)
        report = engine.check_section("1.1", "Some content")
        assert report.reliability_score == 100.0

    def test_no_provider_returns_perfect_score(self):
        engine = FactcheckEngine(provider=None, enabled=True)
        report = engine.check_section("1.1", "Some content")
        assert report.reliability_score == 100.0

    def test_normalize_status(self):
        assert FactcheckEngine._normalize_status("CORROBORÉE") == CORROBORATED
        assert FactcheckEngine._normalize_status("CORROBOREE") == CORROBORATED
        assert FactcheckEngine._normalize_status("PLAUSIBLE") == PLAUSIBLE
        assert FactcheckEngine._normalize_status("NON FONDÉE") == UNFOUNDED
        assert FactcheckEngine._normalize_status("NON FONDEE") == UNFOUNDED
        assert FactcheckEngine._normalize_status("CONTREDITE") == CONTRADICTED
        assert FactcheckEngine._normalize_status("unknown") == PLAUSIBLE

    def test_build_report(self, engine):
        data = {
            "claims": [
                {"id": 1, "text": "Fact 1", "status": "CORROBORÉE", "justification": "OK"},
                {"id": 2, "text": "Fact 2", "status": "PLAUSIBLE", "justification": "Maybe"},
                {"id": 3, "text": "Fact 3", "status": "NON FONDÉE", "justification": "No support"},
            ]
        }
        report = engine._build_report("1.1", data)
        assert report.total_claims == 3
        assert report.status_counts[CORROBORATED] == 1
        assert report.status_counts[PLAUSIBLE] == 1
        assert report.status_counts[UNFOUNDED] == 1
        # 2/3 = 66.7%
        assert abs(report.reliability_score - 66.7) < 0.1

    def test_build_report_empty_claims(self, engine):
        report = engine._build_report("1.1", {"claims": []})
        assert report.total_claims == 0
        assert report.reliability_score == 100.0

    def test_should_correct(self, engine):
        report = FactcheckReport(section_id="1.1", reliability_score=75.0)
        assert engine.should_correct(report) is True
        report.reliability_score = 90.0
        assert engine.should_correct(report) is False

    def test_get_correction_instruction(self, engine):
        report = FactcheckReport(section_id="1.1")
        report.details = [
            ClaimResult(1, "False claim", UNFOUNDED, "No support"),
            ClaimResult(2, "True claim", CORROBORATED, "Confirmed"),
            ClaimResult(3, "Contradicted", CONTRADICTED, "Conflicting"),
        ]
        instruction = engine.get_correction_instruction(report)
        assert "False claim" in instruction
        assert "Contradicted" in instruction
        assert "True claim" not in instruction

    def test_get_correction_instruction_no_problems(self, engine):
        report = FactcheckReport(section_id="1.1")
        report.details = [
            ClaimResult(1, "OK", CORROBORATED),
        ]
        assert engine.get_correction_instruction(report) == ""

    def test_report_serialization(self):
        report = FactcheckReport(
            section_id="1.1",
            total_claims=2,
            status_counts={CORROBORATED: 1, PLAUSIBLE: 1},
            reliability_score=100.0,
        )
        report.details.append(ClaimResult(1, "Claim 1", CORROBORATED, "OK"))
        d = report.to_dict()
        assert d["section_id"] == "1.1"
        report2 = FactcheckReport.from_dict(d)
        assert report2.total_claims == 2

    def test_parse_json_response_valid(self):
        text = '{"claims": [{"id": 1, "text": "test"}]}'
        result = FactcheckEngine._parse_json_response(text)
        assert "claims" in result

    def test_parse_json_response_invalid(self):
        result = FactcheckEngine._parse_json_response("not json")
        assert result == {}

    def test_save_report(self, tmp_path):
        engine = FactcheckEngine(project_dir=tmp_path, enabled=True)
        report = FactcheckReport(section_id="1.1", reliability_score=85.0)
        engine._save_report(report)
        assert (tmp_path / "factcheck" / "1.1.json").exists()
