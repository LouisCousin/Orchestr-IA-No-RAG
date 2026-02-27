"""Test d'intégration : pipeline multi-agents complet sur corpus 3 sections.

Vérifie que le pipeline complet (Architecture → Génération → Vérification →
Évaluation → Correction) fonctionne de bout en bout avec un corpus fictif.
"""

import asyncio
import json
import pytest

from src.core.agent_framework import AgentConfig
from src.core.multi_agent_orchestrator import MultiAgentOrchestrator, GenerationResult
from src.core.orchestrator import ProjectState
from src.core.plan_parser import NormalizedPlan, PlanSection
from src.providers.base import AIResponse


# ── Mock Provider avec réponses réalistes ───────────────────────────────────


class RealisticMockProvider:
    """Provider qui retourne des réponses JSON structurées selon le contexte du prompt."""

    def __init__(self):
        self.name = "mock"
        self._call_count = 0

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        self._call_count += 1

        # Détecter le type d'appel selon le system_prompt
        sp = (system_prompt or "").lower()

        if "architecte" in sp or "dépendances" in sp:
            content = json.dumps({
                "sections": [
                    {"id": "s01", "title": "Introduction", "longueur_cible": 300,
                     "ton": "introductif", "type": "introduction"},
                    {"id": "s02", "title": "Analyse", "longueur_cible": 500,
                     "ton": "analytique", "type": "fond"},
                    {"id": "s03", "title": "Conclusion", "longueur_cible": 200,
                     "ton": "conclusif", "type": "conclusion"},
                ],
                "dependances": {
                    "s01": [],
                    "s02": [],
                    "s03": ["s01", "s02"],
                },
                "zones_risque": [
                    {"section_id": "s02", "description": "Données chiffrées à vérifier"},
                ],
                "system_prompt_global": "Rédige en français professionnel.",
            }, ensure_ascii=False)

        elif "vérificateur" in sp or "factuel" in sp:
            content = json.dumps({
                "verdict": "ok",
                "problemes": [],
                "suggestions": [],
                "score_coherence": 0.85,
            }, ensure_ascii=False)

        elif "évaluateur" in sp or "qualité" in sp:
            content = json.dumps({
                "score_global": 4.2,
                "scores_par_critere": {
                    "pertinence_corpus": 4.5,
                    "precision_factuelle": 4.0,
                    "coherence_interne": 4.1,
                    "qualite_redactionnelle": 4.3,
                    "completude": 4.0,
                    "respect_plan": 4.3,
                },
                "scores_par_section": {"s01": 4.0, "s02": 4.2, "s03": 4.5},
                "sections_a_corriger": [],
                "recommandation": "exporter",
                "commentaire": "Document de bonne qualité.",
            }, ensure_ascii=False)

        elif "correcteur" in sp:
            content = "Contenu corrigé de la section."

        else:
            # Rédacteur par défaut
            section_id = "?"
            if "s01" in prompt.lower() or "introduction" in prompt.lower():
                section_id = "s01"
                content = (
                    "Le présent document a pour objectif de présenter une analyse "
                    "détaillée des données du corpus. Cette introduction pose le cadre "
                    "général de l'étude et définit les objectifs principaux."
                )
            elif "s02" in prompt.lower() or "analyse" in prompt.lower():
                section_id = "s02"
                content = (
                    "L'analyse des données montre des résultats significatifs. "
                    "Les indicateurs clés confirment les tendances observées dans "
                    "le corpus source. Le taux de croissance s'établit à 15%."
                )
            elif "s03" in prompt.lower() or "conclusion" in prompt.lower():
                section_id = "s03"
                content = (
                    "En conclusion, cette analyse a permis de mettre en lumière "
                    "les points essentiels du corpus. Les résultats confirment "
                    "l'hypothèse initiale et ouvrent des perspectives futures."
                )
            else:
                content = "Contenu généré pour la section demandée."

        return AIResponse(
            content=content,
            model=model or "mock-model",
            provider="mock",
            input_tokens=500,
            output_tokens=200,
        )

    def is_available(self):
        return True

    def get_default_model(self):
        return "mock-model"

    def list_models(self):
        return ["mock-model"]


# ── Fixtures ────────────────────────────────────────────────────────────────


SAMPLE_CORPUS_TEXT = """
Le projet d'analyse porte sur les performances financières du premier semestre 2025.

Les données clés sont les suivantes :
- Chiffre d'affaires : 12.5 millions d'euros (+15% vs N-1)
- Marge brute : 45%
- Effectifs : 250 collaborateurs

L'analyse sectorielle montre une progression dans tous les segments d'activité.
Le segment B2B représente 60% du chiffre d'affaires, en croissance de 18%.
Le segment B2C contribue à hauteur de 40%, avec une croissance de 12%.

Les perspectives pour le second semestre sont positives, avec un carnet de commandes
en hausse de 20% par rapport à la même période de l'année précédente.
"""


@pytest.fixture
def project_state():
    state = ProjectState(name="test_integration")
    state.plan = NormalizedPlan(
        sections=[
            PlanSection(id="s01", title="Introduction", level=1,
                        description="Introduction générale"),
            PlanSection(id="s02", title="Analyse des performances", level=1,
                        description="Analyse détaillée des données"),
            PlanSection(id="s03", title="Conclusion", level=1,
                        description="Synthèse et perspectives"),
        ],
        title="Rapport d'analyse",
        objective="Analyser les performances financières du S1 2025",
    )
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


# ── Tests ───────────────────────────────────────────────────────────────────


class TestMultiAgentPipeline:
    def test_full_pipeline(self, project_state):
        """Pipeline complet sur corpus 3 sections."""
        provider = RealisticMockProvider()
        config = AgentConfig.from_config(project_state.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        # Sections générées
        assert isinstance(result, GenerationResult)
        assert len(result.sections) == 3
        assert "s01" in result.sections
        assert "s02" in result.sections
        assert "s03" in result.sections

        # Chaque section a du contenu non vide
        for sid, content in result.sections.items():
            assert len(content) > 0, f"Section {sid} est vide"

    def test_architecture_produced(self, project_state):
        """L'Architecte produit un plan enrichi."""
        provider = RealisticMockProvider()
        config = AgentConfig.from_config(project_state.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        assert result.architecture is not None
        assert "sections" in result.architecture
        assert "dependances" in result.architecture

    def test_verification_reports(self, project_state):
        """Les rapports de vérification sont produits."""
        provider = RealisticMockProvider()
        config = AgentConfig.from_config(project_state.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        assert len(result.verif_reports) > 0
        for sid, report in result.verif_reports.items():
            assert "verdict" in report
            assert "score_coherence" in report

    def test_evaluation_result(self, project_state):
        """L'Évaluateur produit un score global."""
        provider = RealisticMockProvider()
        config = AgentConfig.from_config(project_state.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        assert "score_global" in result.eval_result
        assert result.eval_result["score_global"] > 0

    def test_project_state_updated(self, project_state):
        """Le ProjectState est mis à jour après le pipeline."""
        provider = RealisticMockProvider()
        config = AgentConfig.from_config(project_state.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        # Mettre à jour le state comme le ferait page_generation.py
        project_state.generated_sections = result.sections
        project_state.agent_architecture = result.architecture
        project_state.agent_verif_reports = result.verif_reports
        project_state.agent_eval_result = result.eval_result

        assert len(project_state.generated_sections) == 3
        assert project_state.agent_architecture is not None

    def test_cost_tracking(self, project_state):
        """Les coûts sont suivis."""
        provider = RealisticMockProvider()
        config = AgentConfig.from_config(project_state.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        assert result.total_duration_ms > 0
        assert len(result.token_breakdown) > 0

    def test_timeline_recorded(self, project_state):
        """La timeline des agents est enregistrée."""
        provider = RealisticMockProvider()
        config = AgentConfig.from_config(project_state.config)

        orchestrator = MultiAgentOrchestrator(
            project_state=project_state,
            agent_config=config,
            providers={"mock": provider},
        )

        result = asyncio.run(orchestrator.run())

        assert len(result.agent_timeline) > 0
        # L'architecte est le premier événement
        first = result.agent_timeline[0]
        assert first["agent"] == "architecte"
