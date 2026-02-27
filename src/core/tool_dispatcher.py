"""Dispatcher de tools pour le function calling des agents.

Phase 7 : gère les trois tools du Correcteur (search_corpus, get_section,
           flag_unresolvable) et la boucle de function calling multi-provider.
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional

from src.core.agent_framework import AgentMessage, AgentResult, BaseAgent
from src.core.message_bus import MessageBus
from src.providers.base import AIResponse

logger = logging.getLogger("orchestria")


# ── Définitions des tools (format unifié) ────────────────────────────────────

TOOL_DEFINITIONS = {
    "search_corpus": {
        "description": (
            "Recherche des passages pertinents dans le corpus source. "
            "Retourne des fenêtres de texte autour des occurrences trouvées."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Termes à rechercher dans le corpus",
                "required": True,
            },
            "max_windows": {
                "type": "integer",
                "description": "Nombre de fenêtres retournées (défaut: 5)",
                "required": False,
                "default": 5,
            },
            "window_tokens": {
                "type": "integer",
                "description": "Taille d'une fenêtre en tokens (défaut: 400)",
                "required": False,
                "default": 400,
            },
        },
    },
    "get_section": {
        "description": (
            "Récupère le contenu d'une section déjà générée pour vérifier "
            "la cohérence avec la section en cours de correction."
        ),
        "parameters": {
            "section_id": {
                "type": "string",
                "description": "Identifiant de la section (ex: 's03')",
                "required": True,
            },
        },
    },
    "flag_unresolvable": {
        "description": (
            "Signale qu'une section ne peut pas être corrigée sans information "
            "supplémentaire absente du corpus. Déclenche une notification."
        ),
        "parameters": {
            "section_id": {
                "type": "string",
                "description": "Section concernée",
                "required": True,
            },
            "reason": {
                "type": "string",
                "description": "Description du problème non résolvable",
                "required": True,
            },
        },
    },
}


# ── Fonctions utilitaires ────────────────────────────────────────────────────


def extract_windows(
    corpus_text: str,
    query: str,
    max_windows: int = 5,
    window_tokens: int = 400,
) -> str:
    """Recherche textuelle directe sur le corpus.

    Localise les occurrences des mots clés de la query dans corpus_text
    et retourne N fenêtres de ±window_tokens/2 tokens autour de chaque
    occurrence. Remplacement No-RAG du search_corpus ChromaDB.
    """
    if not corpus_text or not query:
        return "Aucun résultat trouvé."

    # Extraire les mots clés significatifs (> 3 caractères)
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    if not keywords:
        keywords = [query.lower()]

    # Tokeniser le corpus en mots (approximation)
    words = corpus_text.split()
    corpus_lower = corpus_text.lower()

    # Trouver les positions des occurrences
    positions = []
    for keyword in keywords:
        start = 0
        while True:
            idx = corpus_lower.find(keyword, start)
            if idx == -1:
                break
            # Convertir la position caractère en position mot (approximation)
            word_pos = len(corpus_text[:idx].split())
            positions.append(word_pos)
            start = idx + len(keyword)

    if not positions:
        return "Aucun résultat trouvé pour la requête."

    # Dédupliquer et trier les positions
    positions = sorted(set(positions))

    # Extraire les fenêtres
    half_window = window_tokens // 2
    windows = []
    used_ranges = []

    for pos in positions:
        if len(windows) >= max_windows:
            break

        start = max(0, pos - half_window)
        end = min(len(words), pos + half_window)

        # Vérifier le chevauchement avec les fenêtres existantes
        overlaps = False
        for used_start, used_end in used_ranges:
            if start < used_end and end > used_start:
                overlaps = True
                break

        if overlaps:
            continue

        window_text = " ".join(words[start:end])
        windows.append(f"[Fenêtre {len(windows) + 1}]\n{window_text}")
        used_ranges.append((start, end))

    if not windows:
        return "Aucun résultat pertinent trouvé."

    return "\n\n".join(windows)


class ToolDispatcher:
    """Gère l'exécution des tools appelés par le Correcteur."""

    def __init__(self, corpus_text: str, bus: MessageBus):
        self._corpus = corpus_text
        self._bus = bus

    def dispatch(self, tool_name: str, tool_args: dict) -> str:
        """Route l'appel de tool vers la fonction Python correspondante."""
        try:
            if tool_name == "search_corpus":
                return self._search_corpus(
                    query=tool_args.get("query", ""),
                    max_windows=tool_args.get("max_windows", 5),
                    window_tokens=tool_args.get("window_tokens", 400),
                )
            elif tool_name == "get_section":
                return self._get_section(
                    section_id=tool_args.get("section_id", ""),
                )
            elif tool_name == "flag_unresolvable":
                return self._flag_unresolvable(
                    section_id=tool_args.get("section_id", ""),
                    reason=tool_args.get("reason", ""),
                )
            else:
                return f"Tool inconnu : {tool_name}"
        except Exception as e:
            logger.error(f"Erreur dispatch tool {tool_name}: {e}")
            return f"Erreur lors de l'exécution du tool {tool_name}: {e}"

    def _search_corpus(
        self, query: str, max_windows: int = 5, window_tokens: int = 400
    ) -> str:
        return extract_windows(self._corpus, query, max_windows, window_tokens)

    def _get_section(self, section_id: str) -> str:
        return self._bus.get_section_sync(section_id)

    def _flag_unresolvable(self, section_id: str, reason: str) -> str:
        message = AgentMessage(
            sender="correcteur",
            recipient="*",
            type="alert",
            payload={"section_id": section_id, "reason": reason},
            section_id=section_id,
            priority=1,
        )
        # Synchronous publish — run in event loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._bus.publish(message))
            else:
                loop.run_until_complete(self._bus.publish(message))
        except RuntimeError:
            # No event loop — store directly in history
            self._bus._history.append(message)

        return "Alerte transmise à l'utilisateur."

    def get_tool_definitions(self, provider: str) -> list:
        """Retourne les définitions de tools au format du provider cible."""
        if provider == "openai":
            return self._format_openai()
        elif provider == "anthropic":
            return self._format_anthropic()
        elif provider in ("gemini", "google"):
            return self._format_gemini()
        else:
            return self._format_openai()

    def _format_openai(self) -> list:
        """Format OpenAI Chat Completions tools."""
        tools = []
        for name, defn in TOOL_DEFINITIONS.items():
            properties = {}
            required = []
            for param_name, param_info in defn["parameters"].items():
                prop = {"type": param_info["type"], "description": param_info["description"]}
                properties[param_name] = prop
                if param_info.get("required", False):
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": defn["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools

    def _format_anthropic(self) -> list:
        """Format Anthropic Messages API tools."""
        tools = []
        for name, defn in TOOL_DEFINITIONS.items():
            properties = {}
            required = []
            for param_name, param_info in defn["parameters"].items():
                prop = {"type": param_info["type"], "description": param_info["description"]}
                properties[param_name] = prop
                if param_info.get("required", False):
                    required.append(param_name)

            tools.append({
                "name": name,
                "description": defn["description"],
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        return tools

    def _format_gemini(self) -> list:
        """Format Gemini FunctionDeclaration."""
        declarations = []
        for name, defn in TOOL_DEFINITIONS.items():
            properties = {}
            required = []
            for param_name, param_info in defn["parameters"].items():
                gtype = {"string": "STRING", "integer": "INTEGER"}.get(
                    param_info["type"], "STRING"
                )
                properties[param_name] = {
                    "type": gtype,
                    "description": param_info["description"],
                }
                if param_info.get("required", False):
                    required.append(param_name)

            declarations.append({
                "name": name,
                "description": defn["description"],
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": required,
                },
            })
        return declarations

    async def run_agent_with_tools(
        self,
        agent: BaseAgent,
        initial_prompt: str,
        system_prompt: str,
        max_tool_calls: int = 10,
    ) -> tuple[str, int, int]:
        """Gère la boucle de function calling.

        Returns:
            Tuple (final_content, total_input_tokens, total_output_tokens).
        """
        total_input = 0
        total_output = 0
        messages = [{"role": "user", "content": initial_prompt}]
        tool_calls_count = 0

        while tool_calls_count < max_tool_calls:
            response = await agent._call_provider(
                prompt=self._messages_to_prompt(messages),
                system_prompt=system_prompt,
            )
            total_input += response.input_tokens
            total_output += response.output_tokens

            # Extraire les appels de tools du contenu de la réponse
            tool_calls = self._extract_tool_calls(response.content)

            if not tool_calls:
                # Réponse finale sans appel de tool
                return response.content, total_input, total_output

            # Exécuter chaque appel de tool
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_result = self.dispatch(tool_name, tool_args)
                tool_calls_count += 1

                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Résultat de l'appel {tool_name}({json.dumps(tool_args, ensure_ascii=False)}):\n\n"
                        f"{tool_result}\n\n"
                        "Continue la correction en utilisant ce résultat."
                    ),
                })

        # Max tool calls atteint — demander une réponse finale
        response = await agent._call_provider(
            prompt=self._messages_to_prompt(messages) + "\n\nRéponds maintenant avec le contenu corrigé final.",
            system_prompt=system_prompt,
        )
        total_input += response.input_tokens
        total_output += response.output_tokens
        return response.content, total_input, total_output

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convertit une liste de messages en prompt unique."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                parts.append(f"[Assistant précédent]:\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def _extract_tool_calls(self, content: str) -> list[dict]:
        """Extrait les appels de tools depuis le contenu de la réponse.

        Cherche le pattern JSON : {"tool": "name", "arguments": {...}}
        ou des blocs ```json avec tool_call. Déduplique les résultats.
        """
        tool_calls = []
        seen = set()

        def _add(name: str, arguments: dict) -> None:
            key = (name, json.dumps(arguments, sort_keys=True))
            if key not in seen:
                seen.add(key)
                tool_calls.append({"name": name, "arguments": arguments})

        # Pattern 1: Bloc ```json avec structure tool_call (prioritaire)
        code_block_pattern = r'```json\s*(\{.*?\})\s*```'
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "tool" in data and "arguments" in data:
                    tool_name = data["tool"]
                    if tool_name in TOOL_DEFINITIONS:
                        _add(tool_name, data["arguments"])
            except (json.JSONDecodeError, KeyError):
                continue

        # Pattern 2: Bloc JSON explicite avec "tool" et "arguments"
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        for match in re.finditer(json_pattern, content, re.DOTALL):
            try:
                tool_name = match.group(1)
                args = json.loads(match.group(2))
                if tool_name in TOOL_DEFINITIONS:
                    _add(tool_name, args)
            except (json.JSONDecodeError, IndexError):
                continue

        return tool_calls
