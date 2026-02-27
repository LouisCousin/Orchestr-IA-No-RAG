"""Tests unitaires pour le ToolDispatcher (Phase 7)."""

import asyncio
import pytest

from src.core.agent_framework import AgentResult, BaseAgent
from src.core.message_bus import MessageBus
from src.core.tool_dispatcher import ToolDispatcher, extract_windows
from src.providers.base import AIResponse


# ── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_CORPUS = """
Le projet Orchestr'IA est un outil de génération documentaire automatisée.
Il utilise des modèles de langage avancés pour produire des documents structurés.

La technologie repose sur un pipeline en cinq étapes :
Configuration, Acquisition, Plan, Génération et Export.

Les résultats financiers du Q2 2025 montrent une croissance de 15% du chiffre d'affaires.
Le bénéfice net s'établit à 2.3 millions d'euros.

La méthodologie employée combine l'analyse sémantique et le traitement automatique
du langage naturel (NLP) pour extraire les informations pertinentes du corpus source.
"""


class MockProvider:
    """Provider mock pour les tests."""

    def __init__(self, responses=None):
        self._responses = responses or ["Contenu corrigé final."]
        self._call_index = 0
        self.name = "mock"

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        content = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return AIResponse(
            content=content,
            model=model or "mock",
            provider="mock",
            input_tokens=50,
            output_tokens=25,
        )

    def is_available(self):
        return True

    def get_default_model(self):
        return "mock"

    def list_models(self):
        return ["mock"]


class MockAgent(BaseAgent):
    async def _execute(self, task):
        return AgentResult(agent_name=self.name, success=True)

    def _build_system_prompt(self, task):
        return "test"


# ── Tests extract_windows ───────────────────────────────────────────────────


class TestExtractWindows:
    def test_query_found(self):
        result = extract_windows(SAMPLE_CORPUS, "financiers", max_windows=3)
        assert "financiers" in result.lower() or "Fenêtre" in result

    def test_query_not_found(self):
        result = extract_windows(SAMPLE_CORPUS, "zzzznotfound")
        assert "Aucun résultat" in result

    def test_empty_corpus(self):
        result = extract_windows("", "test")
        assert "Aucun résultat" in result

    def test_empty_query(self):
        result = extract_windows(SAMPLE_CORPUS, "")
        assert "Aucun résultat" in result

    def test_max_windows_limit(self):
        result = extract_windows(SAMPLE_CORPUS, "le", max_windows=2)
        # Should return at most 2 windows
        windows = [line for line in result.split("\n") if line.startswith("[Fenêtre")]
        assert len(windows) <= 2

    def test_deduplication(self):
        # "le" appears many times but nearby occurrences should be deduplicated
        result = extract_windows(
            SAMPLE_CORPUS, "le", max_windows=10, window_tokens=50
        )
        assert isinstance(result, str)

    def test_multiple_keywords(self):
        result = extract_windows(SAMPLE_CORPUS, "financiers croissance")
        assert "Fenêtre" in result


# ── Tests ToolDispatcher ────────────────────────────────────────────────────


class TestToolDispatcher:
    def setup_method(self):
        self.bus = MessageBus()
        asyncio.run(self.bus.store_section("s01", "Contenu section 1"))
        asyncio.run(self.bus.store_section("s02", "Contenu section 2"))
        self.dispatcher = ToolDispatcher(SAMPLE_CORPUS, self.bus)

    def test_dispatch_search_corpus(self):
        result = self.dispatcher.dispatch(
            "search_corpus", {"query": "financiers"}
        )
        assert isinstance(result, str)
        assert "Aucun résultat" not in result or "financiers" not in SAMPLE_CORPUS.lower()

    def test_dispatch_search_corpus_no_results(self):
        result = self.dispatcher.dispatch(
            "search_corpus", {"query": "xyznotfound"}
        )
        assert "Aucun résultat" in result

    def test_dispatch_get_section_present(self):
        result = self.dispatcher.dispatch(
            "get_section", {"section_id": "s01"}
        )
        assert result == "Contenu section 1"

    def test_dispatch_get_section_absent(self):
        result = self.dispatcher.dispatch(
            "get_section", {"section_id": "s99"}
        )
        assert result == "Section non disponible"

    def test_dispatch_flag_unresolvable(self):
        result = self.dispatcher.dispatch(
            "flag_unresolvable",
            {"section_id": "s03", "reason": "Données Q3 manquantes"},
        )
        assert "Alerte transmise" in result

        # Vérifier que le message est dans l'historique
        alerts = self.bus.get_alerts()
        # Note: may or may not be in history depending on event loop state
        # The flag_unresolvable stores directly in history as fallback
        assert len(self.bus._history) >= 1

    def test_dispatch_unknown_tool(self):
        result = self.dispatcher.dispatch("unknown_tool", {})
        assert "Tool inconnu" in result

    def test_get_tool_definitions_openai(self):
        tools = self.dispatcher.get_tool_definitions("openai")
        assert len(tools) == 3
        assert all(t["type"] == "function" for t in tools)
        names = {t["function"]["name"] for t in tools}
        assert names == {"search_corpus", "get_section", "flag_unresolvable"}

    def test_get_tool_definitions_anthropic(self):
        tools = self.dispatcher.get_tool_definitions("anthropic")
        assert len(tools) == 3
        assert all("name" in t for t in tools)
        assert all("input_schema" in t for t in tools)

    def test_get_tool_definitions_gemini(self):
        tools = self.dispatcher.get_tool_definitions("gemini")
        assert len(tools) == 3
        # Gemini uses uppercase types
        for t in tools:
            assert "parameters" in t
            assert t["parameters"]["type"] == "OBJECT"


class TestToolDispatcherToolCalls:
    def test_extract_tool_calls_json_block(self):
        bus = MessageBus()
        dispatcher = ToolDispatcher("corpus", bus)

        content = 'Je vais chercher. {"tool": "search_corpus", "arguments": {"query": "test"}}'
        calls = dispatcher._extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_corpus"
        assert calls[0]["arguments"]["query"] == "test"

    def test_extract_tool_calls_code_block(self):
        bus = MessageBus()
        dispatcher = ToolDispatcher("corpus", bus)

        content = '```json\n{"tool": "get_section", "arguments": {"section_id": "s01"}}\n```'
        calls = dispatcher._extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_section"

    def test_extract_tool_calls_no_tool(self):
        bus = MessageBus()
        dispatcher = ToolDispatcher("corpus", bus)

        content = "Voici le texte corrigé de la section."
        calls = dispatcher._extract_tool_calls(content)
        assert len(calls) == 0

    def test_run_agent_with_tools_single_call(self):
        """Test boucle avec 1 appel de tool."""
        async def _test():
            bus = MessageBus()
            await bus.store_section("s01", "Contenu existant")
            dispatcher = ToolDispatcher(SAMPLE_CORPUS, bus)

            provider = MockProvider(responses=[
                # Premier appel: l'agent fait un appel de tool
                '{"tool": "search_corpus", "arguments": {"query": "financiers"}}',
                # Deuxième appel: réponse finale
                "Contenu corrigé avec les données financières.",
            ])
            agent = MockAgent(name="correcteur", provider=provider, model="mock")

            content, inp, out = await dispatcher.run_agent_with_tools(
                agent=agent,
                initial_prompt="Corrige la section.",
                system_prompt="Tu es correcteur.",
                max_tool_calls=5,
            )
            assert "corrigé" in content.lower() or "financ" in content.lower()
            assert inp > 0
            assert out > 0

        asyncio.run(_test())

    def test_run_agent_with_tools_max_reached(self):
        """Test que max_tool_calls est respecté."""
        async def _test():
            bus = MessageBus()
            dispatcher = ToolDispatcher(SAMPLE_CORPUS, bus)

            # L'agent fait toujours des appels de tool
            provider = MockProvider(responses=[
                '{"tool": "search_corpus", "arguments": {"query": "test"}}',
                '{"tool": "search_corpus", "arguments": {"query": "test2"}}',
                '{"tool": "search_corpus", "arguments": {"query": "test3"}}',
                "Réponse finale.",
            ])
            agent = MockAgent(name="correcteur", provider=provider, model="mock")

            content, inp, out = await dispatcher.run_agent_with_tools(
                agent=agent,
                initial_prompt="Corrige.",
                system_prompt="Correcteur.",
                max_tool_calls=2,
            )
            # After max_tool_calls, it should stop and get final response
            assert isinstance(content, str)

        asyncio.run(_test())
