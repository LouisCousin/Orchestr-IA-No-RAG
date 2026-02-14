"""Fournisseur OpenAI pour Orchestr'IA."""

import os
import time
import logging
from typing import Optional

from src.providers.base import AIResponse, BaseProvider

logger = logging.getLogger("orchestria")


class OpenAIProvider(BaseProvider):
    """Fournisseur OpenAI."""

    MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 2.0):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """Génère du contenu via l'API OpenAI avec retry automatique."""
        model = model or self.get_default_model()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = response.choices[0]
                usage = response.usage
                return AIResponse(
                    content=choice.message.content or "",
                    model=model,
                    provider=self.name,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    finish_reason=choice.finish_reason or "",
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(f"Erreur API OpenAI (tentative {attempt + 1}/{self._max_retries + 1}): {e}. Retry dans {delay}s...")
                    time.sleep(delay)

        raise RuntimeError(f"Échec après {self._max_retries + 1} tentatives: {last_error}")

    def is_available(self) -> bool:
        """Vérifie si la clé API est configurée."""
        return bool(self._api_key and self._api_key != "sk-your-openai-api-key-here")

    def get_default_model(self) -> str:
        return "gpt-4o"

    def list_models(self) -> list[str]:
        return self.MODELS.copy()
