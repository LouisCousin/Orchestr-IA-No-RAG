"""Tests unitaires pour glossary_engine.py."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.core.glossary_engine import GlossaryEngine


@pytest.fixture
def engine(tmp_path):
    """Crée un GlossaryEngine avec un répertoire projet."""
    return GlossaryEngine(project_dir=tmp_path, max_terms_per_prompt=5, enabled=True)


@pytest.fixture
def engine_no_dir():
    """Crée un GlossaryEngine sans répertoire."""
    return GlossaryEngine(project_dir=None, enabled=False)


class TestGlossaryEngine:
    def test_add_term(self, engine):
        entry = engine.add_term(
            term="IA",
            definition="Intelligence artificielle",
            abbreviation="IA",
            preferred_form="intelligence artificielle",
            avoid_forms=["AI"],
            domain="technologie",
        )
        assert entry["term"] == "IA"
        assert entry["definition"] == "Intelligence artificielle"
        assert entry["abbreviation"] == "IA"

    def test_add_duplicate_raises(self, engine):
        engine.add_term(term="Test", definition="Def1")
        with pytest.raises(ValueError, match="existe déjà"):
            engine.add_term(term="test", definition="Def2")  # case-insensitive

    def test_get_term(self, engine):
        engine.add_term(term="RAG", definition="Retrieval Augmented Generation")
        result = engine.get_term("rag")  # case-insensitive
        assert result is not None
        assert result["term"] == "RAG"

    def test_get_term_not_found(self, engine):
        assert engine.get_term("nonexistent") is None

    def test_update_term(self, engine):
        engine.add_term(term="Test", definition="Old def")
        result = engine.update_term("Test", definition="New def")
        assert result is not None
        assert result["definition"] == "New def"
        assert result["term"] == "Test"  # key not changed

    def test_update_nonexistent(self, engine):
        assert engine.update_term("Ghost", definition="x") is None

    def test_delete_term(self, engine):
        engine.add_term(term="ToDelete", definition="bye")
        assert engine.delete_term("todelete") is True
        assert engine.get_term("ToDelete") is None

    def test_delete_nonexistent(self, engine):
        assert engine.delete_term("nope") is False

    def test_get_all_terms(self, engine):
        engine.add_term(term="A", definition="def A")
        engine.add_term(term="B", definition="def B")
        terms = engine.get_all_terms()
        assert len(terms) == 2

    def test_persistence(self, tmp_path):
        eng1 = GlossaryEngine(project_dir=tmp_path, enabled=True)
        eng1.add_term(term="Persist", definition="Saved to disk")

        eng2 = GlossaryEngine(project_dir=tmp_path, enabled=True)
        assert eng2.get_term("Persist") is not None

    def test_parse_terms_valid_json(self):
        response = '{"terms": [{"term": "NLP", "definition": "Natural Language Processing", "domain": "IA"}]}'
        terms = GlossaryEngine._parse_terms(response)
        assert len(terms) == 1
        assert terms[0]["term"] == "NLP"
        assert terms[0]["definition"] == "Natural Language Processing"

    def test_parse_terms_invalid_json(self):
        assert GlossaryEngine._parse_terms("not json") == []

    def test_parse_terms_missing_fields(self):
        response = '{"terms": [{"term": "NLP"}, {"term": "IA", "definition": "AI"}]}'
        terms = GlossaryEngine._parse_terms(response)
        assert len(terms) == 1  # Only the one with both term and definition
        assert terms[0]["term"] == "IA"

    def test_parse_terms_with_markdown_wrapper(self):
        response = '```json\n{"terms": [{"term": "T1", "definition": "D1"}]}\n```'
        terms = GlossaryEngine._parse_terms(response)
        assert len(terms) == 1

    def test_apply_generated_terms(self, engine):
        terms = [
            {"term": "A", "definition": "Def A"},
            {"term": "B", "definition": "Def B"},
            {"term": "A", "definition": "Duplicate"},  # duplicate
        ]
        added = engine.apply_generated_terms(terms)
        assert added == 2
        assert len(engine.get_all_terms()) == 2

    def test_get_terms_for_section_relevance(self, engine):
        engine.add_term(term="Machine Learning", definition="ML", domain="IA")
        engine.add_term(term="SQL", definition="Query language", domain="base de données")
        engine.add_term(term="Python", definition="Langage", domain="programmation")

        terms = engine.get_terms_for_section("Introduction au Machine Learning")
        # Machine Learning should be ranked higher
        assert terms[0]["term"] == "Machine Learning"

    def test_get_terms_for_section_respects_max(self, engine):
        for i in range(10):
            engine.add_term(term=f"Term{i}", definition=f"Def{i}")
        terms = engine.get_terms_for_section("Introduction")
        assert len(terms) <= engine.max_terms_per_prompt

    def test_get_terms_for_section_empty_glossary(self, engine):
        assert engine.get_terms_for_section("Section title") == []

    def test_format_for_prompt_with_terms(self, engine):
        terms = [
            {
                "term": "IA",
                "definition": "Intelligence artificielle",
                "preferred_form": "intelligence artificielle",
                "abbreviation": "IA",
                "avoid_forms": ["AI", "A.I."],
            }
        ]
        result = engine.format_for_prompt(terms)
        assert "GLOSSAIRE" in result
        assert "IA" in result
        assert "Forme préférée" in result
        assert "Abréviation" in result
        assert "Éviter" in result

    def test_format_for_prompt_empty(self, engine):
        assert engine.format_for_prompt([]) == ""

    def test_format_for_prompt_minimal(self, engine):
        terms = [{"term": "X", "definition": "Y", "preferred_form": "X"}]
        result = engine.format_for_prompt(terms)
        assert "X : Y" in result
        # preferred_form == term, so should not show "Forme préférée"
        assert "Forme préférée" not in result

    def test_get_chunk_text_dict(self):
        assert GlossaryEngine._get_chunk_text({"text": "hello"}) == "hello"

    def test_get_chunk_text_object(self):
        obj = MagicMock()
        obj.text = "world"
        assert GlossaryEngine._get_chunk_text(obj) == "world"

    def test_get_chunk_text_string(self):
        assert GlossaryEngine._get_chunk_text("plain") == "plain"
