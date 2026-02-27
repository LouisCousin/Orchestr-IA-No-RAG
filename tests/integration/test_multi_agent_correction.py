"""Test d'intégration : correction multi-agents.

Vérifie que le Vérificateur détecte une erreur factuelle et que le
Correcteur la corrige, améliorant le score de la section.
"""

import asyncio
import json
import pytest

from src.core.agent_framework import AgentConfig
from src.core.multi_agent_orchestrator import MultiAgentOrchestrator
from src.core.orchestrator import ProjectState
from src.core.plan_parser import NormalizedPlan, PlanSection
from src.providers.base import AIResponse


# ── Provider qui simule une section fautive nécessitant correction ──────────


class CorrectionMockProvider:
    """Provider qui simule la détection d'erreur et la correction."""

    def __init__(self):
        self.name = "mock"
        self._call_count = 0

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        self._call_count += 1
        sp = (system_prompt or "").lower()

        if "architecte" in sp or "dépendances" in sp:
            content = json.dumps({
                "sections": [
                    {"id": "s01", "title": "Introduction", "longueur_cible": 300,
                     "ton": "introductif", "type": "introduction"},
                    {"id": "s02", "title": "Données fautives", "longueur_cible": 500,
                     "ton": "analytique", "type": "fond"},
                ],
                "dependances": {"s01": [], "s02": []},
                "zones_risque": [
                    {"section_id": "s02", "description": "Données chiffrées"},
                ],
                "system_prompt_global": "Précision factuelle requise.",
            }, ensure_ascii=False)

        elif "vérificateur" in sp or "factuel" in sp:
            if "s02" in prompt or "fautive" in prompt.lower():
                # Le vérificateur détecte une erreur dans s02
                content = json.dumps({
                    "verdict": "erreur",
                    "problemes": [{
                        "type": "FACTUEL",
                        "description": "Le chiffre d'affaires est 12.5M et non 15M",
                        "passage_incrimine": "Le CA s'élève à 15 millions",
                    }],
                    "suggestions": [
                        "Corriger le CA à 12.5 millions d'euros."
                    ],
                    "score_coherence": 0.5,
                })
            else:
                content = json.dumps({
                    "verdict": "ok",
                    "problemes": [],
                    "suggestions": [],
                    "score_coherence": 0.9,
                })

        elif "évaluateur" in sp or "qualité" in sp:
            content = json.dumps({
                "score_global": 3.0,
                "scores_par_critere": {
                    "pertinence_corpus": 4.0,
                    "precision_factuelle": 2.0,
                    "coherence_interne": 3.5,
                    "qualite_redactionnelle": 3.5,
                    "completude": 3.0,
                    "respect_plan": 3.0,
                },
                "scores_par_section": {"s01": 4.0, "s02": 2.5},
                "sections_a_corriger": ["s02"],
                "recommandation": "corriger",
                "commentaire": "La section s02 contient une erreur factuelle.",
            }, ensure_ascii=False)

        elif "correcteur" in sp:
            # Le correcteur produit une version corrigée
            content = (
                "L'analyse des données montre des résultats significatifs. "
                "Le chiffre d'affaires s'établit à 12.5 millions d'euros, "
                "en croissance de 15% par rapport à l'exercice précédent."
            )

        else:
            # Rédacteur
            if "s02" in prompt or "fautive" in prompt.lower():
                # Section avec erreur factuelle délibérée
                content = (
                    "L'analyse montre des résultats encourageants. "
                    "Le CA s'élève à 15 millions d'euros, soit une croissance "
                    "de 20%. Ces chiffres sont erronés par rapport au corpus."
                )
            else:
                content = "Introduction standard du document."

        return AIResponse(
            content=content,
            model=model or "mock",
            provider="mock",
            input_tokens=200,
            output_tokens=100,
        )

    def is_available(self):
        return True

    def get_default_model(self):
        return "mock"

    def list_models(self):
        return ["mock"]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def project_state_correction():
    state = ProjectState(name="test_correction")
    state.plan = NormalizedPlan(
        sections=[
            PlanSection(id="s01", title="Introduction", level=1),
            PlanSection(id="s02", title="Données fautives", level=1),
        ],
        title="Document avec erreur",
        objective="Test de correction automatique",
    )
    state.config = {
        "multi_agent": {
            "enabled": True,
            "max_parallel_writers": 2,
            "max_parallel_verifiers": 2,
            "quality_threshold": 3.5,
            "section_correction_threshold": 3.0,
            "max_correction_passes": 2,
            "max_cost_usd": 10.0,
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


class TestMultiAgentCorrection:
    def test_verifier_detects_error(self, project_state_correction):
        """Le Vérificateur détecte l'erreur factuelle dans s02."""
        provider = CorrectionMockProvider()
        config = AgentConfig.from_config(project_state_correction.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_correction,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        # Vérifier que s02 a un rapport avec une erreur
        assert "s02" in result.verif_reports
        report = result.verif_reports["s02"]
        assert report["verdict"] == "erreur"
        assert len(report["problemes"]) > 0
        assert report["problemes"][0]["type"] == "FACTUEL"

    def test_corrector_fixes_section(self, project_state_correction):
        """Le Correcteur corrige la section identifiée."""
        provider = CorrectionMockProvider()
        config = AgentConfig.from_config(project_state_correction.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_correction,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        # La section s02 devrait avoir été corrigée
        assert "s02" in result.corrections_made
        assert result.corrections_made["s02"] >= 1

        # Le contenu final de s02 devrait être la version corrigée
        assert "12.5 millions" in result.sections.get("s02", "")

    def test_evaluation_triggers_correction(self, project_state_correction):
        """L'Évaluateur identifie correctement les sections à corriger."""
        provider = CorrectionMockProvider()
        config = AgentConfig.from_config(project_state_correction.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_correction,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        assert result.eval_result.get("recommandation") == "corriger"
        assert "s02" in result.eval_result.get("sections_a_corriger", [])

    def test_pipeline_continues_after_correction(self, project_state_correction):
        """Le pipeline produit toutes les sections même après correction."""
        provider = CorrectionMockProvider()
        config = AgentConfig.from_config(project_state_correction.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state_correction,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        # Les deux sections doivent exister
        assert "s01" in result.sections
        assert "s02" in result.sections
        assert len(result.sections["s01"]) > 0
        assert len(result.sections["s02"]) > 0
