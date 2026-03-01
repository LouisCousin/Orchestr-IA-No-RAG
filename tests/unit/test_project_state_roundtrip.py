"""Tests unitaires pour le round-trip ProjectState.to_dict() / from_dict()."""

import pytest

from src.core.orchestrator import ProjectState
from src.core.plan_parser import NormalizedPlan, PlanSection


@pytest.fixture
def full_config():
    """Configuration complète avec toutes les clés."""
    return {
        "model": "claude-sonnet-4-5-20250514",
        "temperature": 0.5,
        "max_tokens": 8192,
        "number_of_passes": 2,
        "default_provider": "anthropic",
        "mode": "manual",
        "checkpoints": {
            "after_plan_validation": True,
            "after_corpus_acquisition": False,
            "after_extraction": False,
            "after_prompt_generation": False,
            "after_generation": True,
            "final_review": True,
        },
        "styling": {
            "primary_color": "#FF0000",
            "secondary_color": "#00FF00",
            "font_title": "Arial",
            "font_body": "Times New Roman",
            "font_size_title": 18,
            "font_size_body": 12,
        },
        "rag_top_k": 10,
        "conditional_generation_enabled": True,
        "target_pages": 25,
        "objective": "Créer un rapport d'analyse complet",
    }


@pytest.fixture
def state_with_plan(full_config):
    """État de projet complet avec plan et sections générées."""
    plan = NormalizedPlan(title="Rapport test", objective="Objectif test")
    plan.sections = [
        PlanSection(id="1", title="Introduction", level=1, page_budget=2.0),
        PlanSection(id="1.1", title="Contexte", level=2, parent_id="1"),
        PlanSection(id="2", title="Analyse", level=1, page_budget=5.0),
    ]

    state = ProjectState(
        name="Projet Test",
        config=full_config,
        plan=plan,
        generated_sections={"1": "Contenu intro", "1.1": "Contenu contexte"},
        section_summaries=["[1] Introduction: résumé", "[1.1] Contexte: résumé"],
        current_step="generation",
        current_section_index=2,
        current_pass=1,
        deferred_sections=["2"],
    )
    return state


class TestProjectStateRoundTrip:
    """Vérifie que to_dict() → from_dict() conserve toutes les données."""

    def test_roundtrip_preserves_name(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.name == "Projet Test"

    def test_roundtrip_preserves_config_model(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["model"] == "claude-sonnet-4-5-20250514"

    def test_roundtrip_preserves_config_temperature(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["temperature"] == 0.5

    def test_roundtrip_preserves_config_max_tokens(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["max_tokens"] == 8192

    def test_roundtrip_preserves_config_number_of_passes(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["number_of_passes"] == 2

    def test_roundtrip_preserves_config_default_provider(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["default_provider"] == "anthropic"

    def test_roundtrip_preserves_config_mode(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["mode"] == "manual"

    def test_roundtrip_preserves_config_checkpoints(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["checkpoints"]["after_plan_validation"] is True
        assert restored.config["checkpoints"]["after_corpus_acquisition"] is False

    def test_roundtrip_preserves_config_styling(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["styling"]["primary_color"] == "#FF0000"
        assert restored.config["styling"]["font_title"] == "Arial"

    def test_roundtrip_preserves_config_rag_top_k(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["rag_top_k"] == 10

    def test_roundtrip_preserves_config_conditional_generation(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.config["conditional_generation_enabled"] is True

    def test_roundtrip_preserves_plan(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.plan is not None
        assert len(restored.plan.sections) == 3
        assert restored.plan.sections[0].title == "Introduction"
        assert restored.plan.sections[0].page_budget == 2.0
        assert restored.plan.sections[1].parent_id == "1"

    def test_roundtrip_preserves_generated_sections(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.generated_sections == {"1": "Contenu intro", "1.1": "Contenu contexte"}

    def test_roundtrip_preserves_section_summaries(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert len(restored.section_summaries) == 2

    def test_roundtrip_preserves_current_step(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.current_step == "generation"

    def test_roundtrip_preserves_current_pass(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.current_pass == 1

    def test_roundtrip_preserves_deferred_sections(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.deferred_sections == ["2"]

    def test_roundtrip_preserves_v4_fields(self, state_with_plan):
        """v4.0 : vérifie que les nouveaux champs survivent au round-trip."""
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.processing_strategy == "standard"
        assert restored.token_stats["total_input_corpus"] == 0
        assert restored.cache_lifecycle["cache_name"] is None

    def test_roundtrip_preserves_timestamps(self, state_with_plan):
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.created_at == state_with_plan.created_at

    def test_roundtrip_empty_state(self):
        """Vérifie le round-trip d'un état minimal."""
        state = ProjectState(name="Vide")
        data = state.to_dict()
        restored = ProjectState.from_dict(data)
        assert restored.name == "Vide"
        assert restored.plan is None
        assert restored.generated_sections == {}
        assert restored.config == {}

    def test_roundtrip_all_config_keys(self, state_with_plan, full_config):
        """Vérifie que TOUTES les clés de configuration survivent au round-trip."""
        data = state_with_plan.to_dict()
        restored = ProjectState.from_dict(data)
        for key in full_config:
            assert key in restored.config, f"Clé manquante après round-trip : {key}"
            assert restored.config[key] == full_config[key], (
                f"Valeur différente pour {key}: {restored.config[key]} != {full_config[key]}"
            )
