"""Tests unitaires pour metadata_overrides.py."""

import pytest
from pathlib import Path
from unittest.mock import patch
from src.core.metadata_overrides import MetadataOverrides, ALLOWED_FIELDS, MIN_YEAR, MAX_YEAR


@pytest.fixture
def overrides(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return MetadataOverrides(project_dir=project_dir)


class TestMetadataOverrides:
    def test_save_and_load_override(self, overrides):
        data = {"title": "Mon article", "year": 2024, "authors": ["Dupont, J."]}
        overrides.save_override("doc_001", data)
        loaded = overrides.load_override("doc_001")
        assert loaded["title"] == "Mon article"
        assert loaded["year"] == 2024
        assert loaded["authors"] == ["Dupont, J."]

    def test_load_nonexistent(self, overrides):
        assert overrides.load_override("nonexistent") == {}

    def test_delete_override(self, overrides):
        overrides.save_override("doc_del", {"title": "Delete me"})
        assert overrides.delete_override("doc_del") is True
        assert overrides.load_override("doc_del") == {}

    def test_delete_nonexistent(self, overrides):
        assert overrides.delete_override("ghost") is False

    def test_has_override(self, overrides):
        assert overrides.has_override("doc_x") is False
        overrides.save_override("doc_x", {"title": "X"})
        assert overrides.has_override("doc_x") is True

    def test_list_overrides(self, overrides):
        overrides.save_override("doc_a", {"title": "A"})
        overrides.save_override("doc_b", {"title": "B"})
        result = overrides.list_overrides()
        assert sorted(result) == ["doc_a", "doc_b"]

    def test_list_overrides_empty(self, overrides):
        assert overrides.list_overrides() == []

    # ── Validation ──

    def test_validate_filters_unknown_fields(self, overrides):
        data = {"title": "Valid", "unknown_field": "ignored", "also_bad": 123}
        result = overrides.save_override("doc_v", data)
        loaded = overrides.load_override("doc_v")
        assert "title" in loaded
        assert "unknown_field" not in loaded
        assert "also_bad" not in loaded

    def test_validate_year_valid(self, overrides):
        overrides.save_override("doc_y", {"year": 2024})
        loaded = overrides.load_override("doc_y")
        assert loaded["year"] == 2024

    def test_validate_year_too_old(self, overrides):
        overrides.save_override("doc_y2", {"year": 1800})
        loaded = overrides.load_override("doc_y2")
        assert "year" not in loaded

    def test_validate_year_too_future(self, overrides):
        overrides.save_override("doc_y3", {"year": MAX_YEAR + 10})
        loaded = overrides.load_override("doc_y3")
        assert "year" not in loaded

    def test_validate_year_non_numeric(self, overrides):
        overrides.save_override("doc_y4", {"year": "not a year"})
        loaded = overrides.load_override("doc_y4")
        assert "year" not in loaded

    def test_validate_year_string_numeric(self, overrides):
        overrides.save_override("doc_y5", {"year": "2020"})
        loaded = overrides.load_override("doc_y5")
        assert loaded["year"] == 2020

    def test_validate_authors_list(self, overrides):
        overrides.save_override("doc_a1", {"authors": ["A", "B"]})
        loaded = overrides.load_override("doc_a1")
        assert loaded["authors"] == ["A", "B"]

    def test_validate_authors_single_string(self, overrides):
        overrides.save_override("doc_a2", {"authors": "Single Author"})
        loaded = overrides.load_override("doc_a2")
        assert loaded["authors"] == ["Single Author"]

    def test_validate_authors_list_non_strings(self, overrides):
        overrides.save_override("doc_a3", {"authors": [123, 456]})
        loaded = overrides.load_override("doc_a3")
        assert "authors" not in loaded

    def test_validate_authors_invalid_type(self, overrides):
        overrides.save_override("doc_a4", {"authors": 42})
        loaded = overrides.load_override("doc_a4")
        assert "authors" not in loaded

    def test_validate_empty_data(self):
        result = MetadataOverrides._validate({})
        assert result == {}

    def test_validate_none_data(self):
        result = MetadataOverrides._validate(None)
        assert result == {}

    # ── Merge metadata ──

    def test_merge_defaults_only(self, overrides):
        result = overrides.merge_metadata("doc_m1")
        assert result["title"] is None
        assert result["doc_type"] == "unknown"

    def test_merge_pdf_data(self, overrides):
        pdf_data = {"title": "PDF Title", "year": 2020}
        result = overrides.merge_metadata("doc_m2", pdf_data=pdf_data)
        assert result["title"] == "PDF Title"
        assert result["year"] == 2020

    def test_merge_grobid_overrides_pdf(self, overrides):
        pdf_data = {"title": "PDF Title", "year": 2020}
        grobid_data = {"title": "GROBID Title", "journal": "Nature"}
        result = overrides.merge_metadata("doc_m3", grobid_data=grobid_data, pdf_data=pdf_data)
        assert result["title"] == "GROBID Title"
        assert result["year"] == 2020  # from PDF (GROBID didn't have it)
        assert result["journal"] == "Nature"

    def test_merge_yaml_overrides_all(self, overrides):
        overrides.save_override("doc_m4", {"title": "Override Title", "year": 2024})
        pdf_data = {"title": "PDF Title", "year": 2020}
        grobid_data = {"title": "GROBID Title", "year": 2022}
        result = overrides.merge_metadata("doc_m4", grobid_data=grobid_data, pdf_data=pdf_data)
        assert result["title"] == "Override Title"
        assert result["year"] == 2024

    def test_merge_grobid_pages_mapping(self, overrides):
        grobid_data = {"pages": "100-120"}
        result = overrides.merge_metadata("doc_m5", grobid_data=grobid_data)
        assert result["pages_range"] == "100-120"

    # ── Path sanitization ──

    def test_doc_id_with_slashes(self, overrides):
        overrides.save_override("path/to/doc", {"title": "Slashed"})
        assert overrides.has_override("path/to/doc")
        loaded = overrides.load_override("path/to/doc")
        assert loaded["title"] == "Slashed"

    # ── Allowed fields constant ──

    def test_allowed_fields(self):
        expected = {"title", "authors", "year", "journal", "volume", "issue",
                    "pages_range", "doi", "publisher", "doc_type"}
        assert ALLOWED_FIELDS == expected
