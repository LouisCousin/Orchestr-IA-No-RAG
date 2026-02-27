"""Bus de messages asynchrone pour la communication inter-agents.

Phase 7 : permet la notification de complétion, la transmission d'alertes,
           le stockage et la récupération de sections générées, et la mise
           à jour du tableau de bord temps réel.
"""

import asyncio
import logging
import threading
import time
from typing import Optional

from src.core.agent_framework import AgentMessage

logger = logging.getLogger("orchestria")


class MessageBus:
    """Bus de messages asynchrone pour la communication inter-agents."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._history: list[AgentMessage] = []
        self._sections: dict[str, str] = {}  # section_id -> contenu
        self._lock: asyncio.Lock = asyncio.Lock()
        self._sync_lock: threading.Lock = threading.Lock()
        self._section_events: dict[str, asyncio.Event] = {}

    async def publish(self, message: AgentMessage) -> None:
        """Publie un message dans la file du destinataire.

        Si recipient == "*", diffuse à tous les agents abonnés.
        """
        async with self._lock:
            self._history.append(message)

        if message.recipient == "*":
            for name, queue in self._queues.items():
                if name != message.sender:
                    await queue.put(message)
        elif message.recipient in self._queues:
            await self._queues[message.recipient].put(message)
        else:
            logger.warning(
                f"MessageBus: destinataire inconnu '{message.recipient}' "
                f"(message de {message.sender})"
            )

    async def subscribe(self, agent_name: str) -> asyncio.Queue:
        """Crée et retourne la file de messages d'un agent."""
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()
        return self._queues[agent_name]

    async def store_section(self, section_id: str, content: str) -> None:
        """Stocke le contenu d'une section générée."""
        async with self._lock:
            self._sections[section_id] = content
            if section_id in self._section_events:
                self._section_events[section_id].set()

    async def get_section(self, section_id: str) -> Optional[str]:
        """Récupère le contenu d'une section par son identifiant."""
        return self._sections.get(section_id)

    def get_section_sync(self, section_id: str) -> str:
        """Version synchrone de get_section (pour les tools du Correcteur)."""
        return self._sections.get(section_id, "Section non disponible")

    def get_all_sections(self) -> dict[str, str]:
        """Retourne toutes les sections stockées."""
        return dict(self._sections)

    def get_history(self) -> list[AgentMessage]:
        """Retourne l'historique complet des messages."""
        with self._sync_lock:
            return list(self._history)

    async def wait_for(
        self,
        section_id: str,
        timeout_s: int = 300,
    ) -> Optional[str]:
        """Attend la publication d'une section spécifique avec timeout."""
        if section_id in self._sections:
            return self._sections[section_id]

        if section_id not in self._section_events:
            self._section_events[section_id] = asyncio.Event()

        try:
            await asyncio.wait_for(
                self._section_events[section_id].wait(),
                timeout=timeout_s,
            )
            return self._sections.get(section_id)
        except asyncio.TimeoutError:
            logger.warning(
                f"MessageBus: timeout en attendant la section {section_id}"
            )
            return None

    def store_alert_sync(self, message: AgentMessage) -> None:
        """Stocke un message d'alerte de manière synchrone (hors event loop)."""
        with self._sync_lock:
            self._history.append(message)

    def get_alerts(self) -> list[AgentMessage]:
        """Retourne toutes les alertes émises."""
        with self._sync_lock:
            return [m for m in self._history if m.type == "alert"]

    def reset(self) -> None:
        """Réinitialise le bus (pour les tests)."""
        self._queues.clear()
        self._history.clear()
        self._sections.clear()
        self._section_events.clear()
