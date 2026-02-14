"""Fournisseur Anthropic (Claude) pour Orchestr'IA."""

import os
import time
import logging
from typing import Optional

from src.providers.base import AIResponse, BaseProvider

logger = logging.getLogger("orchestria")


class AnthropicProvider(BaseProvider):
    """Fournisseur Anthropic (Claude 4.5 / Opus 4.6)."""

    MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-5-20250514",
        "claude-haiku-35-20241022",
    ]

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 2.0):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "anthropic"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """Génère du contenu via l'API Anthropic avec retry automatique."""
        model = model or self.get_default_model()

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()

                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = client.messages.create(**kwargs)

                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text

                input_tokens = response.usage.input_tokens if response.usage else 0
                output_tokens = response.usage.output_tokens if response.usage else 0

                return AIResponse(
                    content=content,
                    model=model,
                    provider=self.name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    finish_reason=response.stop_reason or "",
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        f"Erreur API Anthropic (tentative {attempt + 1}/{self._max_retries + 1}): {e}. "
                        f"Retry dans {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(f"Échec après {self._max_retries + 1} tentatives: {last_error}")

    def is_available(self) -> bool:
        """Vérifie si la clé API est configurée."""
        return bool(self._api_key and self._api_key != "sk-ant-your-anthropic-api-key-here")

    def get_default_model(self) -> str:
        return "claude-sonnet-4-5-20250514"

    def list_models(self) -> list[str]:
        return self.MODELS.copy()
