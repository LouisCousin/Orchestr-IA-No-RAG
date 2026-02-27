"""Test d'intégration : parallélisme effectif du pipeline multi-agents.

Vérifie que les sections indépendantes sont bien générées en parallèle
et qu'aucune section n'est perdue.
"""

import asyncio
import time
import pytest

from src.core.agent_framework import AgentConfig
from src.core.multi_agent_orchestrator import MultiAgentOrchestrator
from src.core.orchestrator import ProjectState
from src.core.plan_parser import NormalizedPlan, PlanSection
from src.providers.base import AIResponse


# ── Provider avec délai simulé ──────────────────────────────────────────────


class SlowMockProvider:
    """Provider qui simule un temps de réponse pour mesurer le parallélisme."""

    def __init__(self, delay_s=0.1):
        self._delay = delay_s
        self.name = "mock"
        self._call_count = 0

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        import json as _json
        self._call_count += 1
        # Simuler un délai réseau
        time.sleep(self._delay)

        sp = (system_prompt or "").lower()

        if "architecte" in sp or "dépendances" in sp:
            sections = []
            deps = {}
            for i in range(1, 9):
                sid = f"s{i:02d}"
                sections.append({
                    "id": sid, "title": f"Section {i}",
                    "longueur_cible": 300, "ton": "analytique", "type": "fond",
                })
                deps[sid] = []  # Toutes indépendantes

            content = _json.dumps({
                "sections": sections,
                "dependances": deps,
                "zones_risque": [],
                "system_prompt_global": "Test.",
            }, ensure_ascii=False)

        elif "vérificateur" in sp or "factuel" in sp:
            content = _json.dumps({
                "verdict": "ok", "problemes": [],
                "suggestions": [], "score_coherence": 0.9,
            })

        elif "évaluateur" in sp or "qualité" in sp:
            content = _json.dumps({
                "score_global": 4.0,
                "scores_par_critere": {
                    "pertinence_corpus": 4.0, "precision_factuelle": 4.0,
                    "coherence_interne": 4.0, "qualite_redactionnelle": 4.0,
                    "completude": 4.0, "respect_plan": 4.0,
                },
                "sections_a_corriger": [],
                "recommandation": "exporter",
                "commentaire": "OK.",
            })

        else:
            content = f"Contenu généré pour la section demandée. Appel #{self._call_count}."

        return AIResponse(
            content=content,
            model=model or "mock",
            provider="mock",
            input_tokens=100,
            output_tokens=50,
        )

    def is_available(self):
        return True

    def get_default_model(self):
        return "mock"

    def list_models(self):
        return ["mock"]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def project_state_8sections():
    state = ProjectState(name="test_parallel")
    sections = [
        PlanSection(id=f"s{i:02d}", title=f"Section {i}", level=1)
        for i in range(1, 9)
    ]
    state.plan = NormalizedPlan(
        sections=sections,
        title="Document parallèle",
        objective="Test parallélisme",
    )
    state.config = {
        "multi_agent": {
            "enabled": True,
            "max_parallel_writers": 4,
            "max_parallel_verifiers": 4,
            "quality_threshold": 3.0,
            "section_correction_threshold": 2.5,
            "max_correction_passes": 0,
            "max_cost_usd": 50.0,
            "agents": {
                "architecte": {"provider": "mock", "model": "mock", "timeout_s": 30},
                "redacteur": {"provider": "mock", "model": "mock", "timeout_s": 30},
                "verificateur": {"provider": "mock", "model": "mock", "timeout_s": 30},
                "evaluateur": {"provider": "mock", "model": "mock", "timeout_s": 30},
                "correcteur": {"provider": "mock", "model": "mock", "timeout_s": 30},
            },
        },
    }
    return state


# ── Tests ───────────────────────────────────────────────────────────────────


class TestParallelGeneration:
    def test_all_sections_generated(self, project_state_8sections):
        """8 sections indépendantes sont toutes générées."""
        provider = SlowMockProvider(delay_s=0.05)
        config = AgentConfig.from_config(project_state_8sections.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_8sections,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        assert len(result.sections) == 8
        for i in range(1, 9):
            sid = f"s{i:02d}"
            assert sid in result.sections, f"Section {sid} manquante"
            assert len(result.sections[sid]) > 0

    def test_no_section_lost(self, project_state_8sections):
        """Aucune section n'est perdue avec le parallélisme."""
        provider = SlowMockProvider(delay_s=0.02)
        config = AgentConfig.from_config(project_state_8sections.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_8sections,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        expected_ids = {f"s{i:02d}" for i in range(1, 9)}
        actual_ids = set(result.sections.keys())
        assert expected_ids == actual_ids

    def test_parallel_faster_than_sequential(self, project_state_8sections):
        """Le mode parallèle est plus rapide qu'un traitement strictement séquentiel."""
        delay = 0.1
        provider = SlowMockProvider(delay_s=delay)
        config = AgentConfig.from_config(project_state_8sections.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_8sections,
            agent_config=config,
            providers={"mock": provider},
        )

        start = time.time()
        result = asyncio.run(orchestrator.run())
        elapsed = time.time() - start

        # Durée séquentielle minimale théorique : 8 sections × delay × 3 agents
        # (rédacteur + vérificateur + architecte + évaluateur ≈ 8*3*0.1 = 2.4s)
        # En parallèle (max_parallel=4), devrait être < 2× la durée séquentielle
        # Utilisons un seuil raisonnable
        sequential_min = 8 * delay  # Juste les rédacteurs séquentiellement
        assert elapsed < sequential_min * 2, (
            f"Durée {elapsed:.2f}s trop lente (séquentiel min: {sequential_min:.2f}s)"
        )

    def test_sections_in_message_bus(self, project_state_8sections):
        """Toutes les sections sont stockées dans le MessageBus."""
        provider = SlowMockProvider(delay_s=0.02)
        config = AgentConfig.from_config(project_state_8sections.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_8sections,
            agent_config=config,
            providers={"mock": provider},
        )

        asyncio.run(orchestrator.run())

        bus_sections = orchestrator.bus.get_all_sections()
        assert len(bus_sections) == 8
