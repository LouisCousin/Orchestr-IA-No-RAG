"""Framework d'agents pour l'orchestration multi-agents.

Phase 7 : dataclasses de base (AgentMessage, AgentResult, AgentState),
           classe abstraite BaseAgent dont héritent les 5 agents spécialisés.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.providers.base import AIResponse, BaseProvider

logger = logging.getLogger("orchestria")


# ── Dataclasses de base ─────────────────────────────────────────────────────


@dataclass
class AgentMessage:
    """Message échangé entre agents via le MessageBus."""

    id: str = ""
    sender: str = ""
    recipient: str = ""  # "*" = broadcast
    type: str = ""  # "task" | "result" | "alert" | "query"
    payload: dict = field(default_factory=dict)
    timestamp: float = 0.0
    section_id: Optional[str] = None
    priority: int = 0  # 0 = normal, 1 = urgent

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "section_id": self.section_id,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMessage":
        return cls(
            id=data.get("id", ""),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            type=data.get("type", ""),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", 0.0),
            section_id=data.get("section_id"),
            priority=data.get("priority", 0),
        )


@dataclass
class AgentResult:
    """Résultat produit par un agent après exécution."""

    agent_name: str = ""
    section_id: Optional[str] = None
    success: bool = True
    content: Optional[str] = None
    structured_data: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: int = 0
    token_input: int = 0
    token_output: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "section_id": self.section_id,
            "success": self.success,
            "content": self.content,
            "structured_data": self.structured_data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "token_input": self.token_input,
            "token_output": self.token_output,
            "cost_usd": self.cost_usd,
        }


@dataclass
class AgentState:
    """État courant d'un agent (pour le tableau de bord UI)."""

    name: str = ""
    status: str = "idle"  # "idle" | "running" | "done" | "error"
    current_task: Optional[str] = None
    progress: float = 0.0  # 0.0 - 1.0
    last_updated: float = 0.0
    results_count: int = 0
    errors_count: int = 0

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "current_task": self.current_task,
            "progress": self.progress,
            "last_updated": self.last_updated,
            "results_count": self.results_count,
            "errors_count": self.errors_count,
        }


# ── Classe abstraite BaseAgent ──────────────────────────────────────────────


class BaseAgent:
    """Classe abstraite dont héritent les 5 agents spécialisés.

    Gère les retries, timeouts, mise à jour de l'état et appels provider.
    """

    def __init__(
        self,
        name: str,
        provider: BaseProvider,
        model: str,
        max_retries: int = 2,
        timeout_s: int = 120,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.name = name
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.state = AgentState(name=name, status="idle")
        self._bus = None  # Injecté par l'orchestrateur

    def set_bus(self, bus) -> None:
        """Injecte le MessageBus (appelé par l'orchestrateur)."""
        self._bus = bus

    async def run(self, task: dict) -> AgentResult:
        """Point d'entrée principal avec gestion des retries et du timeout."""
        self._update_state("running", task.get("description", "Traitement en cours"))

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._execute(task),
                    timeout=self.timeout_s,
                )
                result.duration_ms = max(1, int((time.time() - start_time) * 1000))
                self._update_state("done", progress=1.0)
                self.state.results_count += 1
                return result
            except asyncio.TimeoutError:
                last_error = f"Timeout après {self.timeout_s}s (tentative {attempt + 1})"
                logger.warning(f"[{self.name}] {last_error}")
            except Exception as e:
                last_error = f"{type(e).__name__}: {e} (tentative {attempt + 1})"
                logger.warning(f"[{self.name}] {last_error}")

            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)

        duration_ms = int((time.time() - start_time) * 1000)
        self._update_state("error")
        self.state.errors_count += 1
        return AgentResult(
            agent_name=self.name,
            success=False,
            error=last_error,
            duration_ms=duration_ms,
        )

    async def _execute(self, task: dict) -> AgentResult:
        """Implémenté par chaque agent concret."""
        raise NotImplementedError

    def _build_system_prompt(self, task: dict) -> str:
        """Construit le system prompt spécifique à l'agent et à la tâche."""
        raise NotImplementedError

    async def _call_provider(
        self,
        prompt: str,
        system_prompt: str,
    ) -> AIResponse:
        """Appelle le provider via run_in_executor (le provider est synchrone)."""
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )
        return response

    def _update_state(
        self,
        status: str,
        task: Optional[str] = None,
        progress: float = 0.0,
    ) -> None:
        """Met à jour AgentState et notifie le MessageBus."""
        self.state.status = status
        if task is not None:
            self.state.current_task = task
        self.state.progress = progress
        self.state.last_updated = time.time()


# ── Configuration des agents ────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Configuration globale du pipeline multi-agents."""

    enabled: bool = False
    max_parallel_writers: int = 4
    max_parallel_verifiers: int = 4
    quality_threshold: float = 3.5
    section_correction_threshold: float = 3.0
    max_correction_passes: int = 2
    max_cost_usd: float = 5.0
    agents: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "AgentConfig":
        """Extrait la configuration multi-agents depuis config/default.yaml."""
        ma = config.get("multi_agent", {})
        return cls(
            enabled=ma.get("enabled", False),
            max_parallel_writers=ma.get("max_parallel_writers", 4),
            max_parallel_verifiers=ma.get("max_parallel_verifiers", 4),
            quality_threshold=ma.get("quality_threshold", 3.5),
            section_correction_threshold=ma.get("section_correction_threshold", 3.0),
            max_correction_passes=ma.get("max_correction_passes", 2),
            max_cost_usd=ma.get("max_cost_usd", 5.0),
            agents=ma.get("agents", {}),
        )

    def get_agent_config(self, agent_name: str) -> dict:
        """Retourne la configuration d'un agent spécifique."""
        return self.agents.get(agent_name, {})
