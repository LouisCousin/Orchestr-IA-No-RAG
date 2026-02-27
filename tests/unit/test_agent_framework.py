"""Tests unitaires pour le framework d'agents (Phase 7)."""

import asyncio
import time
import pytest

from src.core.agent_framework import (
    AgentConfig,
    AgentMessage,
    AgentResult,
    AgentState,
    BaseAgent,
)
from src.providers.base import AIResponse


# ── Fixtures ────────────────────────────────────────────────────────────────


class MockProvider:
    """Provider mock pour les tests."""

    def __init__(self, response_content="Test response", fail=False):
        self._content = response_content
        self._fail = fail
        self.name = "mock"
        self.call_count = 0

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        self.call_count += 1
        if self._fail:
            raise RuntimeError("Provider error")
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


class ConcreteAgent(BaseAgent):
    """Agent concret pour les tests."""

    async def _execute(self, task):
        response = await self._call_provider("test prompt", "test system")
        return AgentResult(
            agent_name=self.name,
            success=True,
            content=response.content,
            token_input=response.input_tokens,
            token_output=response.output_tokens,
        )

    def _build_system_prompt(self, task):
        return "test system prompt"


class SlowAgent(BaseAgent):
    """Agent qui prend trop de temps (pour tester le timeout)."""

    async def _execute(self, task):
        await asyncio.sleep(10)
        return AgentResult(agent_name=self.name, success=True)

    def _build_system_prompt(self, task):
        return "slow"


class FailingAgent(BaseAgent):
    """Agent qui échoue (pour tester les retries)."""

    def __init__(self, *args, fail_count=2, **kwargs):
        super().__init__(*args, **kwargs)
        self._fail_count = fail_count
        self._attempts = 0

    async def _execute(self, task):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise ValueError(f"Erreur simulée (tentative {self._attempts})")
        return AgentResult(
            agent_name=self.name,
            success=True,
            content="Success after retry",
        )

    def _build_system_prompt(self, task):
        return "failing"


# ── Tests AgentMessage ──────────────────────────────────────────────────────


class TestAgentMessage:
    def test_creation_auto_id(self):
        msg = AgentMessage(sender="a", recipient="b", type="task")
        assert msg.id  # UUID généré automatiquement
        assert msg.sender == "a"
        assert msg.recipient == "b"
        assert msg.type == "task"
        assert msg.timestamp > 0

    def test_creation_with_id(self):
        msg = AgentMessage(id="custom-id", sender="a", recipient="b", type="task")
        assert msg.id == "custom-id"

    def test_serialization_roundtrip(self):
        msg = AgentMessage(
            sender="architecte",
            recipient="*",
            type="result",
            payload={"key": "value"},
            section_id="s01",
            priority=1,
        )
        d = msg.to_dict()
        restored = AgentMessage.from_dict(d)
        assert restored.sender == msg.sender
        assert restored.recipient == msg.recipient
        assert restored.type == msg.type
        assert restored.payload == msg.payload
        assert restored.section_id == msg.section_id
        assert restored.priority == msg.priority

    def test_default_values(self):
        msg = AgentMessage()
        assert msg.id  # Auto-generated
        assert msg.sender == ""
        assert msg.recipient == ""
        assert msg.priority == 0
        assert msg.section_id is None


# ── Tests AgentResult ───────────────────────────────────────────────────────


class TestAgentResult:
    def test_creation(self):
        result = AgentResult(
            agent_name="redacteur",
            section_id="s01",
            success=True,
            content="texte",
            token_input=100,
            token_output=50,
        )
        assert result.agent_name == "redacteur"
        assert result.success is True
        assert result.content == "texte"

    def test_to_dict(self):
        result = AgentResult(agent_name="test", success=True, cost_usd=1.5)
        d = result.to_dict()
        assert d["agent_name"] == "test"
        assert d["success"] is True
        assert d["cost_usd"] == 1.5

    def test_error_result(self):
        result = AgentResult(
            agent_name="test",
            success=False,
            error="Something failed",
        )
        assert result.success is False
        assert result.error == "Something failed"
        assert result.content is None


# ── Tests AgentState ────────────────────────────────────────────────────────


class TestAgentState:
    def test_initial_state(self):
        state = AgentState(name="architecte")
        assert state.name == "architecte"
        assert state.status == "idle"
        assert state.progress == 0.0
        assert state.last_updated > 0

    def test_transitions(self):
        state = AgentState(name="test")
        assert state.status == "idle"

        state.status = "running"
        assert state.status == "running"

        state.status = "done"
        assert state.status == "done"

        state.status = "error"
        assert state.status == "error"

    def test_to_dict(self):
        state = AgentState(name="test", status="running", progress=0.5)
        d = state.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "running"
        assert d["progress"] == 0.5


# ── Tests BaseAgent ─────────────────────────────────────────────────────────


class TestBaseAgent:
    def test_creation(self):
        provider = MockProvider()
        agent = ConcreteAgent(name="test", provider=provider, model="mock-model")
        assert agent.name == "test"
        assert agent.state.status == "idle"

    def test_run_success(self):
        provider = MockProvider(response_content="Generated content")
        agent = ConcreteAgent(name="test", provider=provider, model="mock-model")
        result = asyncio.run(agent.run({"description": "test task"}))
        assert result.success is True
        assert result.content == "Generated content"
        assert result.duration_ms > 0
        assert agent.state.status == "done"
        assert agent.state.results_count == 1

    def test_run_timeout(self):
        provider = MockProvider()
        agent = SlowAgent(
            name="slow", provider=provider, model="mock",
            timeout_s=1, max_retries=0,
        )
        result = asyncio.run(agent.run({"description": "slow task"}))
        assert result.success is False
        assert "Timeout" in result.error
        assert agent.state.status == "error"
        assert agent.state.errors_count == 1

    def test_run_retry_success(self):
        provider = MockProvider()
        agent = FailingAgent(
            name="failing", provider=provider, model="mock",
            max_retries=2, fail_count=2,
        )
        result = asyncio.run(agent.run({"description": "retry task"}))
        assert result.success is True
        assert result.content == "Success after retry"
        assert agent._attempts == 3  # 2 failures + 1 success

    def test_run_retry_exhausted(self):
        provider = MockProvider()
        agent = FailingAgent(
            name="failing", provider=provider, model="mock",
            max_retries=1, fail_count=5,
        )
        result = asyncio.run(agent.run({"description": "failing task"}))
        assert result.success is False
        assert "Erreur simulée" in result.error

    def test_set_bus(self):
        provider = MockProvider()
        agent = ConcreteAgent(name="test", provider=provider, model="mock")
        assert agent._bus is None
        agent.set_bus("mock_bus")
        assert agent._bus == "mock_bus"


# ── Tests AgentConfig ───────────────────────────────────────────────────────


class TestAgentConfig:
    def test_from_config_default(self):
        config = {}
        ac = AgentConfig.from_config(config)
        assert ac.enabled is False
        assert ac.max_parallel_writers == 4
        assert ac.quality_threshold == 3.5

    def test_from_config_with_values(self):
        config = {
            "multi_agent": {
                "enabled": True,
                "max_parallel_writers": 8,
                "quality_threshold": 4.0,
                "agents": {
                    "architecte": {
                        "provider": "anthropic",
                        "model": "claude-opus-4-5",
                    },
                },
            },
        }
        ac = AgentConfig.from_config(config)
        assert ac.enabled is True
        assert ac.max_parallel_writers == 8
        assert ac.quality_threshold == 4.0

    def test_get_agent_config(self):
        ac = AgentConfig(
            agents={
                "architecte": {"provider": "anthropic", "model": "test"},
            }
        )
        cfg = ac.get_agent_config("architecte")
        assert cfg["provider"] == "anthropic"

        cfg_missing = ac.get_agent_config("unknown")
        assert cfg_missing == {}
