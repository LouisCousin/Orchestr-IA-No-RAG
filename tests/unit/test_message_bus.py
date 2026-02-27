"""Tests unitaires pour le MessageBus (Phase 7)."""

import asyncio
import pytest

from src.core.agent_framework import AgentMessage
from src.core.message_bus import MessageBus


# ── Helpers ─────────────────────────────────────────────────────────────────


def run(coro):
    """Helper pour exécuter une coroutine dans les tests."""
    return asyncio.run(coro)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestMessageBus:
    def test_subscribe_creates_queue(self):
        async def _test():
            bus = MessageBus()
            queue = await bus.subscribe("agent_a")
            assert queue is not None
            assert "agent_a" in bus._queues

        run(_test())

    def test_publish_single_agent(self):
        async def _test():
            bus = MessageBus()
            await bus.subscribe("agent_b")

            msg = AgentMessage(
                sender="agent_a", recipient="agent_b", type="task",
                payload={"test": True},
            )
            await bus.publish(msg)

            queue = bus._queues["agent_b"]
            received = await asyncio.wait_for(queue.get(), timeout=1)
            assert received.sender == "agent_a"
            assert received.payload == {"test": True}

        run(_test())

    def test_broadcast(self):
        async def _test():
            bus = MessageBus()
            await bus.subscribe("agent_a")
            await bus.subscribe("agent_b")
            await bus.subscribe("agent_c")

            msg = AgentMessage(
                sender="agent_a", recipient="*", type="alert",
                payload={"urgent": True},
            )
            await bus.publish(msg)

            # agent_b et agent_c reçoivent, pas agent_a (l'émetteur)
            assert not bus._queues["agent_a"].empty() is False or bus._queues["agent_a"].qsize() == 0
            assert bus._queues["agent_b"].qsize() == 1
            assert bus._queues["agent_c"].qsize() == 1

        run(_test())

    def test_store_and_get_section(self):
        async def _test():
            bus = MessageBus()
            await bus.store_section("s01", "Contenu section 1")
            result = await bus.get_section("s01")
            assert result == "Contenu section 1"

        run(_test())

    def test_get_section_missing(self):
        async def _test():
            bus = MessageBus()
            result = await bus.get_section("inexistant")
            assert result is None

        run(_test())

    def test_get_section_sync(self):
        bus = MessageBus()
        asyncio.run(bus.store_section("s02", "Contenu section 2"))
        assert bus.get_section_sync("s02") == "Contenu section 2"
        assert bus.get_section_sync("inexistant") == "Section non disponible"

    def test_wait_for_section(self):
        async def _test():
            bus = MessageBus()

            async def store_later():
                await asyncio.sleep(0.1)
                await bus.store_section("s03", "Contenu différé")

            asyncio.create_task(store_later())
            result = await bus.wait_for("s03", timeout_s=5)
            assert result == "Contenu différé"

        run(_test())

    def test_wait_for_section_timeout(self):
        async def _test():
            bus = MessageBus()
            result = await bus.wait_for("inexistant", timeout_s=0.1)
            assert result is None

        run(_test())

    def test_wait_for_already_stored(self):
        async def _test():
            bus = MessageBus()
            await bus.store_section("s04", "Déjà là")
            result = await bus.wait_for("s04", timeout_s=1)
            assert result == "Déjà là"

        run(_test())

    def test_history(self):
        async def _test():
            bus = MessageBus()
            await bus.subscribe("agent_a")

            msg1 = AgentMessage(sender="x", recipient="agent_a", type="task")
            msg2 = AgentMessage(sender="y", recipient="agent_a", type="result")
            await bus.publish(msg1)
            await bus.publish(msg2)

            history = bus.get_history()
            assert len(history) == 2
            assert history[0].sender == "x"
            assert history[1].sender == "y"

        run(_test())

    def test_get_alerts(self):
        async def _test():
            bus = MessageBus()
            await bus.subscribe("agent_a")

            msg1 = AgentMessage(sender="x", recipient="agent_a", type="task")
            msg2 = AgentMessage(sender="y", recipient="agent_a", type="alert",
                                payload={"reason": "test"})
            await bus.publish(msg1)
            await bus.publish(msg2)

            alerts = bus.get_alerts()
            assert len(alerts) == 1
            assert alerts[0].type == "alert"

        run(_test())

    def test_concurrent_store_section(self):
        """Vérifie que 4 writers parallèles ne perdent pas de sections."""
        async def _test():
            bus = MessageBus()

            async def store(sid, content):
                await bus.store_section(sid, content)

            tasks = [
                store(f"s{i:02d}", f"Contenu section {i}")
                for i in range(1, 5)
            ]
            await asyncio.gather(*tasks)

            all_sections = bus.get_all_sections()
            assert len(all_sections) == 4
            for i in range(1, 5):
                assert f"s{i:02d}" in all_sections

        run(_test())

    def test_reset(self):
        async def _test():
            bus = MessageBus()
            await bus.subscribe("agent_a")
            await bus.store_section("s01", "test")
            msg = AgentMessage(sender="x", recipient="agent_a", type="task")
            await bus.publish(msg)

            bus.reset()
            assert len(bus._queues) == 0
            assert len(bus._history) == 0
            assert len(bus._sections) == 0

        run(_test())

    def test_get_all_sections(self):
        async def _test():
            bus = MessageBus()
            await bus.store_section("s01", "A")
            await bus.store_section("s02", "B")

            all_s = bus.get_all_sections()
            assert all_s == {"s01": "A", "s02": "B"}

        run(_test())
