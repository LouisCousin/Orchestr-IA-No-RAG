"""Interface commune pour les fournisseurs d'IA."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AIResponse:
    """Réponse d'un fournisseur d'IA."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    raw_response: Optional[dict] = field(default=None, repr=False)


class BaseProvider(ABC):
    """Interface commune pour tous les fournisseurs d'IA."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du fournisseur."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """Génère du contenu à partir d'un prompt."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le fournisseur est configuré et disponible."""
        ...

    @abstractmethod
    def get_default_model(self) -> str:
        """Retourne le modèle par défaut du fournisseur."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Liste les modèles disponibles pour ce fournisseur."""
        ...
