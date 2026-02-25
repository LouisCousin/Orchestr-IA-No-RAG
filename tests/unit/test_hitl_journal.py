"""Tests unitaires pour hitl_journal.py."""

import pytest
from pathlib import Path
from src.core.hitl_journal import HITLJournal


@pytest.fixture
def journal(tmp_path):
    path = tmp_path / "config" / "hitl_journal.json"
    return HITLJournal(journal_path=path)


class TestHITLJournal:
    def test_log_intervention(self, journal):
        entry = journal.log_intervention(
            project_name="test_project",
            checkpoint_type="GENERATION_REVIEW",
            intervention_type="modify",
            section_id="1.1",
            original_content="original text",
            modified_content="modified text",
            delta_summary="Changed tone",
        )
        assert entry["id"] == 1
        assert entry["project_name"] == "test_project"
        assert entry["intervention_type"] == "modify"
        assert entry["section_id"] == "1.1"
        assert "timestamp" in entry

    def test_multiple_entries_increment_id(self, journal):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        entry2 = journal.log_intervention("p1", "GENERATION_REVIEW", "modify")
        assert entry2["id"] == 2

    def test_get_interventions_all(self, journal):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        journal.log_intervention("p2", "GENERATION_REVIEW", "modify")
        assert len(journal.get_interventions()) == 2

    def test_get_interventions_by_project(self, journal):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        journal.log_intervention("p2", "GENERATION_REVIEW", "modify")
        results = journal.get_interventions(project="p1")
        assert len(results) == 1
        assert results[0]["project_name"] == "p1"

    def test_get_interventions_by_type(self, journal):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        journal.log_intervention("p1", "GENERATION_REVIEW", "modify")
        journal.log_intervention("p1", "FINAL_REVIEW", "accept")
        results = journal.get_interventions(intervention_type="accept")
        assert len(results) == 2

    def test_get_interventions_by_section(self, journal):
        journal.log_intervention("p1", "GENERATION_REVIEW", "modify", section_id="1.1")
        journal.log_intervention("p1", "GENERATION_REVIEW", "accept", section_id="1.2")
        results = journal.get_interventions(section="1.1")
        assert len(results) == 1

    def test_get_statistics_empty(self, journal):
        stats = journal.get_statistics()
        assert stats["total"] == 0
        assert stats["modification_rate"] == 0.0

    def test_get_statistics(self, journal):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        journal.log_intervention("p1", "GENERATION_REVIEW", "modify")
        journal.log_intervention("p1", "FINAL_REVIEW", "reject")
        stats = journal.get_statistics()
        assert stats["total"] == 3
        assert stats["by_type"]["accept"] == 1
        assert stats["by_type"]["modify"] == 1

    def test_get_statistics_by_project(self, journal):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        journal.log_intervention("p2", "GENERATION_REVIEW", "modify")
        stats = journal.get_statistics(project="p1")
        assert stats["total"] == 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "config" / "hitl_journal.json"
        j1 = HITLJournal(journal_path=path)
        j1.log_intervention("p1", "PLAN_VALIDATION", "accept")
        # Load fresh
        j2 = HITLJournal(journal_path=path)
        assert len(j2.entries) == 1
        assert j2.entries[0]["project_name"] == "p1"

    def test_export_to_excel(self, journal, tmp_path):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        journal.log_intervention("p2", "GENERATION_REVIEW", "modify")
        filepath = tmp_path / "export.xlsx"
        result = journal.export_to_excel(filepath)
        assert result.exists()

    def test_empty_journal_export(self, journal, tmp_path):
        filepath = tmp_path / "empty.xlsx"
        result = journal.export_to_excel(filepath)
        assert result.exists()

    def test_entries_property(self, journal):
        journal.log_intervention("p1", "PLAN_VALIDATION", "accept")
        entries = journal.entries
        assert len(entries) == 1
        # Ensure it's a copy
        entries.clear()
        assert len(journal.entries) == 1

    def test_content_truncation(self, journal):
        long_text = "x" * 1000
        entry = journal.log_intervention(
            "p1", "GENERATION_REVIEW", "modify",
            original_content=long_text,
        )
        assert len(entry["original_content"]) == 500
