"""Tests unitaires pour persona_engine.py."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.core.persona_engine import PersonaEngine, EXPERTISE_LEVELS, REGISTERS


@pytest.fixture
def engine(tmp_path):
    return PersonaEngine(project_dir=tmp_path, enabled=True)


@pytest.fixture
def engine_disabled():
    return PersonaEngine(project_dir=None, enabled=False)


class TestPersonaEngine:
    def test_create_persona(self, engine):
        pid = engine.create(
            name="Expert technique",
            profile="Ingénieur senior en informatique",
            expertise_level="expert",
            register="technique",
        )
        assert pid is not None
        p = engine.get(pid)
        assert p["name"] == "Expert technique"
        assert p["expertise_level"] == "expert"
        assert p["register"] == "technique"

    def test_first_persona_is_primary(self, engine):
        pid = engine.create(name="First", profile="First persona")
        primary = engine.get_primary()
        assert primary["id"] == pid

    def test_create_invalid_expertise_defaults(self, engine):
        pid = engine.create(name="Bad", profile="Test", expertise_level="invalid")
        p = engine.get(pid)
        assert p["expertise_level"] == "intermédiaire"

    def test_create_invalid_register_defaults(self, engine):
        pid = engine.create(name="Bad", profile="Test", register="invalid")
        p = engine.get(pid)
        assert p["register"] == "formel"

    def test_get_nonexistent(self, engine):
        assert engine.get("nonexistent") is None

    def test_update_persona(self, engine):
        pid = engine.create(name="Original", profile="Old")
        result = engine.update(pid, profile="New", expectations="Updated")
        assert result["profile"] == "New"
        assert result["expectations"] == "Updated"

    def test_update_nonexistent(self, engine):
        assert engine.update("nonexistent", name="x") is None

    def test_delete_persona(self, engine):
        pid = engine.create(name="ToDelete", profile="bye")
        assert engine.delete(pid) is True
        assert engine.get(pid) is None

    def test_delete_primary_promotes_next(self, engine):
        pid1 = engine.create(name="First", profile="a")
        pid2 = engine.create(name="Second", profile="b")
        engine.delete(pid1)
        primary = engine.get_primary()
        assert primary["id"] == pid2

    def test_delete_cleans_section_assignments(self, engine):
        pid = engine.create(name="Assigned", profile="a")
        engine.assign_to_section("sec_1", pid)
        engine.delete(pid)
        # Section should no longer have an assigned persona
        result = engine.get_persona_for_section("sec_1")
        assert result is None  # No personas left

    def test_delete_nonexistent(self, engine):
        assert engine.delete("nonexistent") is False

    def test_list_personas(self, engine):
        engine.create(name="A", profile="a")
        engine.create(name="B", profile="b")
        assert len(engine.list_personas()) == 2

    def test_set_primary(self, engine):
        pid1 = engine.create(name="First", profile="a")
        pid2 = engine.create(name="Second", profile="b")
        assert engine.set_primary(pid2) is True
        assert engine.get_primary()["id"] == pid2

    def test_set_primary_nonexistent(self, engine):
        assert engine.set_primary("nonexistent") is False

    def test_get_primary_empty(self, engine):
        assert engine.get_primary() is None

    def test_assign_to_section(self, engine):
        pid1 = engine.create(name="General", profile="a")
        pid2 = engine.create(name="Specific", profile="b")
        engine.assign_to_section("sec_1", pid2)
        result = engine.get_persona_for_section("sec_1")
        assert result["id"] == pid2

    def test_section_falls_back_to_primary(self, engine):
        pid = engine.create(name="Primary", profile="a")
        result = engine.get_persona_for_section("sec_no_assign")
        assert result["id"] == pid

    def test_format_for_prompt_enabled(self, engine):
        pid = engine.create(
            name="Reader",
            profile="Gestionnaire de projet",
            expertise_level="intermédiaire",
            register="formel",
            expectations="Des recommandations actionables",
            sensitivities="Budget sensible",
            preferred_formats=["tableaux", "listes"],
        )
        result = engine.format_for_prompt()
        assert "Gestionnaire de projet" in result
        assert "intermédiaire" in result
        assert "formel" in result
        assert "recommandations actionables" in result
        assert "Budget sensible" in result
        assert "tableaux" in result

    def test_format_for_prompt_disabled(self, engine_disabled):
        assert engine_disabled.format_for_prompt() == ""

    def test_format_for_prompt_no_personas(self, engine):
        assert engine.format_for_prompt() == ""

    def test_format_for_prompt_specific_persona(self, engine):
        engine.create(name="Default", profile="Default prof")
        persona = {
            "profile": "Custom persona",
            "expertise_level": "expert",
            "register": "technique",
        }
        result = engine.format_for_prompt(persona=persona)
        assert "Custom persona" in result
        assert "expert" in result

    def test_parse_personas_valid(self):
        response = '{"personas": [{"name": "P1", "profile": "Prof1"}, {"name": "P2", "profile": "Prof2"}]}'
        result = PersonaEngine._parse_personas(response)
        assert len(result) == 2
        assert result[0]["name"] == "P1"

    def test_parse_personas_invalid_json(self):
        assert PersonaEngine._parse_personas("not json") == []

    def test_parse_personas_with_markdown(self):
        response = '```json\n{"personas": [{"name": "P1"}]}\n```'
        result = PersonaEngine._parse_personas(response)
        assert len(result) == 1

    def test_persistence(self, tmp_path):
        eng1 = PersonaEngine(project_dir=tmp_path, enabled=True)
        pid = eng1.create(name="Saved", profile="Persisted")
        eng1.assign_to_section("s1", pid)

        eng2 = PersonaEngine(project_dir=tmp_path, enabled=True)
        assert eng2.get(pid) is not None
        assert eng2.get_persona_for_section("s1")["id"] == pid
