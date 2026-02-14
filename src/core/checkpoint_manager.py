"""Gestion des checkpoints HITL (Human-In-The-Loop)."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("orchestria")


class CheckpointType(str, Enum):
    PLAN_VALIDATION = "after_plan_validation"
    CORPUS_ACQUISITION = "after_corpus_acquisition"
    EXTRACTION = "after_extraction"
    PROMPT_GENERATION = "after_prompt_generation"
    GENERATION = "after_generation"
    FINAL_REVIEW = "final_review"


@dataclass
class CheckpointResult:
    """Résultat d'une intervention au checkpoint."""
    checkpoint_type: str
    action: str  # "approved", "modified", "rejected", "skipped"
    timestamp: str = ""
    section_id: Optional[str] = None
    original_content: Optional[str] = None
    modified_content: Optional[str] = None
    user_comment: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CheckpointConfig:
    """Configuration des checkpoints activés."""
    after_plan_validation: bool = True
    after_corpus_acquisition: bool = False
    after_extraction: bool = False
    after_prompt_generation: bool = False
    after_generation: bool = False
    final_review: bool = True

    def is_enabled(self, checkpoint_type: str) -> bool:
        return getattr(self, checkpoint_type, False)

    def to_dict(self) -> dict:
        return {
            "after_plan_validation": self.after_plan_validation,
            "after_corpus_acquisition": self.after_corpus_acquisition,
            "after_extraction": self.after_extraction,
            "after_prompt_generation": self.after_prompt_generation,
            "after_generation": self.after_generation,
            "final_review": self.final_review,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointConfig":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class CheckpointManager:
    """Gère les checkpoints du pipeline en mode manuel."""

    def __init__(self, config: Optional[CheckpointConfig] = None):
        self.config = config or CheckpointConfig()
        self.history: list[CheckpointResult] = []
        self._pending_checkpoint: Optional[dict] = None

    def should_pause(self, checkpoint_type: str) -> bool:
        """Vérifie si le pipeline doit s'arrêter à ce checkpoint."""
        return self.config.is_enabled(checkpoint_type)

    def create_checkpoint(
        self,
        checkpoint_type: str,
        content: Any,
        section_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Crée un checkpoint qui sera présenté à l'utilisateur.

        Retourne les données du checkpoint si activé, None sinon.
        """
        if not self.should_pause(checkpoint_type):
            return None

        checkpoint_data = {
            "type": checkpoint_type,
            "content": content,
            "section_id": section_id,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self._pending_checkpoint = checkpoint_data
        logger.info(f"Checkpoint atteint : {checkpoint_type}" + (f" (section {section_id})" if section_id else ""))
        return checkpoint_data

    def resolve_checkpoint(
        self,
        action: str,
        modified_content: Optional[str] = None,
        user_comment: str = "",
    ) -> CheckpointResult:
        """Résout le checkpoint en cours avec la décision de l'utilisateur."""
        if not self._pending_checkpoint:
            raise RuntimeError("Aucun checkpoint en attente")

        result = CheckpointResult(
            checkpoint_type=self._pending_checkpoint["type"],
            action=action,
            section_id=self._pending_checkpoint.get("section_id"),
            original_content=str(self._pending_checkpoint.get("content", ""))[:500],
            modified_content=modified_content,
            user_comment=user_comment,
        )
        self.history.append(result)
        self._pending_checkpoint = None
        return result

    @property
    def has_pending(self) -> bool:
        return self._pending_checkpoint is not None

    @property
    def pending_checkpoint(self) -> Optional[dict]:
        return self._pending_checkpoint

    def get_history(self) -> list[dict]:
        return [
            {
                "type": r.checkpoint_type,
                "action": r.action,
                "timestamp": r.timestamp,
                "section_id": r.section_id,
                "comment": r.user_comment,
            }
            for r in self.history
        ]
