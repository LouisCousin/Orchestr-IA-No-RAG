"""Tests unitaires pour persistent_instructions.py."""

import pytest
from pathlib import Path
from src.core.persistent_instructions import (
    PersistentInstructions,
    CATEGORIES,
    CATEGORY_LABELS,
    LEVELS,
)


@pytest.fixture
def pi(tmp_path):
    return PersistentInstructions(project_dir=tmp_path)


@pytest.fixture
def pi_no_dir():
    return PersistentInstructions(project_dir=None)


class TestPersistentInstructions:
    # ── Project level ──

    def test_set_and_get_project_instruction(self, pi):
        pi.set_project_instruction("style_ton", "Adopte un ton formel.")
        result = pi.get_project_instructions()
        assert result["style_ton"] == "Adopte un ton formel."

    def test_set_project_invalid_category(self, pi):
        pi.set_project_instruction("invalid_cat", "text")
        assert pi.get_project_instructions() == {}

    # ── Context level ──

    def test_set_and_get_context_instruction(self, pi):
        pi.set_context_instruction("introduction", "contenu", "Inclure des données chiffrées.")
        result = pi.get_context_instructions("introduction")
        assert result["contenu"] == "Inclure des données chiffrées."

    def test_get_context_nonexistent(self, pi):
        assert pi.get_context_instructions("nonexistent") == {}

    def test_set_context_invalid_category(self, pi):
        pi.set_context_instruction("ctx", "invalid_cat", "text")
        assert pi.get_context_instructions("ctx") == {}

    # ── Section level ──

    def test_set_and_get_section_instruction(self, pi):
        pi.set_section_instruction("sec_1.1", "structure", "Utiliser des listes.")
        result = pi.get_section_instructions("sec_1.1")
        assert result["structure"] == "Utiliser des listes."

    def test_get_section_nonexistent(self, pi):
        assert pi.get_section_instructions("nonexistent") == {}

    def test_set_section_invalid_category(self, pi):
        pi.set_section_instruction("sec_1", "invalid_cat", "text")
        assert pi.get_section_instructions("sec_1") == {}

    # ── Context assignment ──

    def test_assign_section_to_context(self, pi):
        pi.set_context_instruction("body", "style_ton", "Ton neutre.")
        pi.assign_section_to_context("sec_2", "body")
        contexts = pi.list_contexts()
        assert "body" in contexts

    # ── Hierarchical resolution ──

    def test_resolution_project_only(self, pi):
        pi.set_project_instruction("style_ton", "Ton formel.")
        result = pi.get_instructions("sec_any")
        assert "[Style et ton] Ton formel." in result

    def test_resolution_context_overrides_project(self, pi):
        pi.set_project_instruction("style_ton", "Ton formel.")
        pi.set_context_instruction("intro", "style_ton", "Ton engageant.")
        pi.assign_section_to_context("sec_1", "intro")

        result = pi.get_instructions("sec_1")
        assert "Ton engageant." in result
        assert "Ton formel." not in result

    def test_resolution_section_overrides_all(self, pi):
        pi.set_project_instruction("style_ton", "Ton formel.")
        pi.set_context_instruction("intro", "style_ton", "Ton engageant.")
        pi.assign_section_to_context("sec_1", "intro")
        pi.set_section_instruction("sec_1", "style_ton", "Ton didactique.")

        result = pi.get_instructions("sec_1")
        assert "Ton didactique." in result
        assert "Ton engageant." not in result
        assert "Ton formel." not in result

    def test_resolution_multiple_categories(self, pi):
        pi.set_project_instruction("style_ton", "Ton formel.")
        pi.set_project_instruction("contenu", "Inclure des sources.")
        pi.set_section_instruction("sec_1", "structure", "Utiliser des tableaux.")

        result = pi.get_instructions("sec_1")
        assert "[Style et ton]" in result
        assert "[Contenu obligatoire/interdit]" in result
        assert "[Structure]" in result

    def test_resolution_empty(self, pi):
        result = pi.get_instructions("sec_any")
        assert result == ""

    def test_resolution_preserves_category_order(self, pi):
        # CATEGORIES order: style_ton, contenu, structure, domaine_metier
        pi.set_project_instruction("domaine_metier", "Domaine éducation.")
        pi.set_project_instruction("style_ton", "Ton formel.")

        result = pi.get_instructions("sec_1")
        style_pos = result.find("[Style et ton]")
        domain_pos = result.find("[Domaine métier]")
        assert style_pos < domain_pos

    # ── Conflict detection ──

    def test_detect_conflicts_none(self, pi):
        pi.set_project_instruction("style_ton", "Same.")
        conflicts = pi.detect_conflicts("sec_1")
        assert conflicts == []

    def test_detect_conflicts_project_vs_context(self, pi):
        pi.set_project_instruction("style_ton", "Formel.")
        pi.set_context_instruction("intro", "style_ton", "Engageant.")
        pi.assign_section_to_context("sec_1", "intro")

        conflicts = pi.detect_conflicts("sec_1")
        assert len(conflicts) == 1
        assert conflicts[0]["category"] == "style_ton"
        assert conflicts[0]["project_text"] == "Formel."
        assert conflicts[0]["context_text"] == "Engageant."

    def test_detect_conflicts_all_levels(self, pi):
        pi.set_project_instruction("contenu", "Version projet.")
        pi.set_context_instruction("ctx", "contenu", "Version contexte.")
        pi.assign_section_to_context("sec_1", "ctx")
        pi.set_section_instruction("sec_1", "contenu", "Version section.")

        conflicts = pi.detect_conflicts("sec_1")
        assert len(conflicts) == 1
        c = conflicts[0]
        assert c["category"] == "contenu"
        assert c["resolution"] == "Version section."

    def test_detect_conflicts_same_values_no_conflict(self, pi):
        pi.set_project_instruction("style_ton", "Identical.")
        pi.set_context_instruction("ctx", "style_ton", "Identical.")
        pi.assign_section_to_context("sec_1", "ctx")

        conflicts = pi.detect_conflicts("sec_1")
        assert conflicts == []

    # ── Utilities ──

    def test_is_configured_false(self, pi):
        assert pi.is_configured() is False

    def test_is_configured_project(self, pi):
        pi.set_project_instruction("style_ton", "Test.")
        assert pi.is_configured() is True

    def test_is_configured_context(self, pi):
        pi.set_context_instruction("ctx", "style_ton", "Test.")
        assert pi.is_configured() is True

    def test_is_configured_section(self, pi):
        pi.set_section_instruction("s1", "style_ton", "Test.")
        assert pi.is_configured() is True

    def test_list_contexts(self, pi):
        pi.set_context_instruction("intro", "style_ton", "A.")
        pi.set_context_instruction("body", "style_ton", "B.")
        contexts = pi.list_contexts()
        assert sorted(contexts) == ["body", "intro"]

    def test_to_dict_from_dict(self, pi):
        pi.set_project_instruction("style_ton", "Formel.")
        pi.set_context_instruction("ctx", "contenu", "Sources.")
        pi.set_section_instruction("s1", "structure", "Listes.")
        pi.assign_section_to_context("s1", "ctx")

        data = pi.to_dict()
        restored = PersistentInstructions.from_dict(data)
        assert restored.get_project_instructions() == pi.get_project_instructions()
        assert restored.get_instructions("s1") == pi.get_instructions("s1")

    def test_persistence(self, tmp_path):
        pi1 = PersistentInstructions(project_dir=tmp_path)
        pi1.set_project_instruction("style_ton", "Persisted.")

        pi2 = PersistentInstructions(project_dir=tmp_path)
        assert pi2.get_project_instructions()["style_ton"] == "Persisted."

    def test_categories_constant(self):
        assert "style_ton" in CATEGORIES
        assert "contenu" in CATEGORIES
        assert "structure" in CATEGORIES
        assert "domaine_metier" in CATEGORIES

    def test_levels_constant(self):
        assert "project" in LEVELS
        assert "context" in LEVELS
        assert "section" in LEVELS
