"""Tests unitaires pour le MultiAgentOrchestrator (Phase 7 + Phase 8)."""

import asyncio
import pytest

from src.core.agent_framework import AgentConfig, AgentMessage, AgentState
from src.core.multi_agent_orchestrator import (
    GenerationResult,
    MultiAgentOrchestrator,
    get_descendants,
)
from src.core.orchestrator import ProjectState
from src.core.plan_parser import NormalizedPlan, PlanSection
from src.providers.base import AIResponse


# ── Fixtures ────────────────────────────────────────────────────────────────


class MockProvider:
    """Provider mock pour les tests."""

    def __init__(self, response_content="Generated content"):
        self._content = response_content
        self.name = "mock"

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        return AIResponse(
            content=self._content,
            model=model or "mock-model",
            provider="mock",
            input_tokens=100,
            output_tokens=50,
        )

    def is_available(self):
        return True

    def get_default_model(self):
        return "mock-model"

    def list_models(self):
        return ["mock-model"]


def make_plan(n_sections=3):
    """Crée un plan test avec N sections."""
    sections = [
        PlanSection(
            id=f"s{i:02d}",
            title=f"Section {i}",
            level=1,
            description=f"Description section {i}",
        )
        for i in range(1, n_sections + 1)
    ]
    return NormalizedPlan(
        sections=sections,
        title="Document test",
        objective="Objectif test",
    )


def make_state(n_sections=3):
    """Crée un ProjectState test."""
    state = ProjectState(name="test_project")
    state.plan = make_plan(n_sections)
    state.config = {
        "multi_agent": {
            "enabled": True,
            "max_parallel_writers": 2,
            "max_parallel_verifiers": 2,
            "quality_threshold": 3.5,
            "section_correction_threshold": 3.0,
            "max_correction_passes": 1,
            "max_cost_usd": 10.0,
            "agents": {
                "architecte": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "redacteur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "verificateur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "evaluateur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "correcteur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
            },
        },
    }
    return state


def make_agent_config(state):
    return AgentConfig.from_config(state.config)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestGenerationResult:
    def test_to_dict(self):
        result = GenerationResult(
            sections={"s01": "Content 1"},
            total_cost_usd=1.2345,
        )
        d = result.to_dict()
        assert d["sections"] == {"s01": "Content 1"}
        assert d["total_cost_usd"] == 1.2345

    def test_empty_result(self):
        result = GenerationResult()
        d = result.to_dict()
        assert d["sections"] == {}
        assert d["total_cost_usd"] == 0.0


class TestMultiAgentOrchestratorInit:
    def test_initialization(self):
        state = make_state()
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        assert orch.bus is not None
        assert len(orch.agents) > 0
        assert orch.is_done() is False

    def test_get_agent_states(self):
        state = make_state()
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        states = orch.get_agent_states()
        assert len(states) > 0
        assert all(isinstance(s, AgentState) for s in states)
        assert all(s.status == "idle" for s in states)

    def test_get_current_metrics(self):
        state = make_state()
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        metrics = orch.get_current_metrics()
        assert "elapsed_ms" in metrics
        assert "sections_generated" in metrics
        assert metrics["sections_generated"] == 0


class TestEstimatePipelineCost:
    def test_basic_estimate(self):
        state = make_state()
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        estimate = orch.estimate_pipeline_cost(
            corpus_tokens=50000,
            section_count=10,
        )

        assert "estimated_usd" in estimate
        assert "total_input_tokens" in estimate
        assert "total_output_tokens" in estimate
        assert "within_budget" in estimate
        assert "token_breakdown" in estimate

        assert estimate["estimated_usd"] > 0
        assert estimate["total_input_tokens"] > 0

    def test_budget_exceeded(self):
        state = make_state()
        state.config["multi_agent"]["max_cost_usd"] = 0.01  # Very low budget
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        estimate = orch.estimate_pipeline_cost(
            corpus_tokens=100000,
            section_count=20,
        )

        assert estimate["within_budget"] is False


class TestMultiAgentOrchestratorFallback:
    def test_default_architecture(self):
        state = make_state(3)
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        arch = orch._default_architecture()
        assert "sections" in arch
        assert "dependances" in arch
        assert len(arch["sections"]) == 3
        # All sections should have no dependencies by default
        for sid, deps in arch["dependances"].items():
            assert isinstance(deps, list)

    def test_provider_fallback(self):
        """Si le provider configuré n'existe pas, fallback sur le premier disponible."""
        state = make_state()
        # Configure architecte with non-existent provider
        state.config["multi_agent"]["agents"]["architecte"]["provider"] = "nonexistent"
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        # L'agent architecte devrait exister avec le fallback
        assert "architecte" in orch.agents


# ── Phase 8 : Context Cache ──────────────────────────────────────────────


class TestContextCacheInit:
    """Phase 8 : intégration du Context Cache Gemini."""

    def test_cache_id_none_by_default(self):
        """Le cache_id est None si caching non configuré."""
        state = make_state()
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        assert orch._cache_id is None

    def test_cache_not_initialized_without_google_provider(self):
        """Le cache n'est pas initialisé si pas de provider Google."""
        state = make_state()
        state.config["gemini"] = {"caching_enabled": True}
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        cache_id = orch._init_context_cache()
        assert cache_id is None

    def test_cache_not_initialized_when_disabled(self):
        """Le cache n'est pas initialisé si caching_enabled=false."""
        state = make_state()
        state.config["gemini"] = {"caching_enabled": False}
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        cache_id = orch._init_context_cache()
        assert cache_id is None

    def test_task_data_includes_cache_id(self):
        """Vérifie que task_data contient cache_id (ou None) pour le Rédacteur."""
        state = make_state()
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )
        # cache_id doit être None car pas de provider Google
        assert orch._cache_id is None


# ── Phase 8 : HITL (workflow scindé) ─────────────────────────────────────


class TestHITLWorkflow:
    """Phase 8 : scission du workflow Architecte / Génération."""

    def test_run_architect_phase_sets_status(self):
        """run_architect_phase doit mettre le statut à waiting_for_architect_validation."""
        state = make_state()
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        # L'Architecte utilise le MockProvider qui ne retourne pas de structured_data,
        # donc il devrait retourner l'architecture par défaut.
        architecture = asyncio.run(orch.run_architect_phase())

        assert architecture is not None
        assert "sections" in architecture
        assert "dependances" in architecture
        assert state.current_step == "waiting_for_architect_validation"
        assert state.agent_architecture is not None

    def test_default_architecture_has_all_sections(self):
        """L'architecture par défaut contient toutes les sections du plan."""
        state = make_state(5)
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        arch = orch._default_architecture()
        assert len(arch["sections"]) == 5
        assert len(arch["dependances"]) == 5


# ── Phase 8 : Circuit Breaker ────────────────────────────────────────────


class TestCircuitBreaker:
    """Phase 8 : annulation en cascade via get_descendants."""

    def test_cancel_descendants_populates_generated(self):
        """_cancel_descendants marque les descendants comme annulés."""
        state = make_state(3)
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )
        # Simuler un DAG linéaire : s01 → s02 → s03
        orch._dag = {"s01": [], "s02": ["s01"], "s03": ["s02"]}

        completed = {"s01"}
        generated = {"s01": "{{GENERATION_FAILED}}"}
        cancelled = set()
        pending_tasks = {}
        in_progress = set()

        orch._cancel_descendants(
            "s01", completed, generated, cancelled, pending_tasks, in_progress,
        )

        # s02 et s03 doivent être annulés
        assert "s02" in cancelled
        assert "s03" in cancelled
        assert "{{CANCELLED_DEPENDENCY_FAILED}}" in generated.get("s02", "")
        assert "{{CANCELLED_DEPENDENCY_FAILED}}" in generated.get("s03", "")
        # Les sections annulées doivent être dans completed
        assert "s02" in completed
        assert "s03" in completed

    def test_cancel_descendants_publishes_alerts(self):
        """Les alertes sont publiées sur le MessageBus pour chaque section annulée."""
        state = make_state(3)
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )
        orch._dag = {"s01": [], "s02": ["s01"], "s03": ["s01"]}

        completed = {"s01"}
        generated = {}
        cancelled = set()

        orch._cancel_descendants(
            "s01", completed, generated, cancelled, {}, set(),
        )

        alerts = orch.bus.get_alerts()
        # 2 alertes (s02 et s03)
        assert len(alerts) == 2
        alert_sids = {a.section_id for a in alerts}
        assert alert_sids == {"s02", "s03"}

    def test_circuit_breaker_does_not_affect_independent_sections(self):
        """Les sections indépendantes ne sont pas touchées par le Circuit Breaker."""
        state = make_state(4)
        config = make_agent_config(state)
        providers = {"mock": MockProvider()}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )
        # s01 → s02, s03 indépendant, s04 dépend de s03
        orch._dag = {
            "s01": [], "s02": ["s01"],
            "s03": [], "s04": ["s03"],
        }

        completed = {"s01"}
        generated = {}
        cancelled = set()

        orch._cancel_descendants(
            "s01", completed, generated, cancelled, {}, set(),
        )

        # Seul s02 doit être annulé
        assert "s02" in cancelled
        assert "s03" not in cancelled
        assert "s04" not in cancelled
