"""Fournisseur Google Gemini pour Orchestr'IA."""

import os
import time
import logging
from typing import Optional

from src.providers.base import AIResponse, BaseProvider

logger = logging.getLogger("orchestria")


class GeminiProvider(BaseProvider):
    """Fournisseur Google Gemini 3."""

    MODELS = [
        "gemini-3.0-pro",
        "gemini-3.0-flash",
    ]

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 2.0):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "google"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """Génère du contenu via l'API Google Gemini avec retry automatique."""
        from google.genai import types

        model = model or self.get_default_model()

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                if system_prompt:
                    config.system_instruction = system_prompt

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                content = response.text or ""

                # Extraire les tokens depuis usage_metadata
                input_tokens = 0
                output_tokens = 0
                if response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                finish_reason = ""
                if response.candidates and response.candidates[0].finish_reason:
                    finish_reason = str(response.candidates[0].finish_reason)

                return AIResponse(
                    content=content,
                    model=model,
                    provider=self.name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    finish_reason=finish_reason,
                )
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        f"Erreur API Gemini (tentative {attempt + 1}/{self._max_retries + 1}): {e}. "
                        f"Retry dans {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(f"Échec après {self._max_retries + 1} tentatives: {last_error}")

    def is_available(self) -> bool:
        """Vérifie si la clé API est configurée."""
        return bool(self._api_key and self._api_key != "your-google-api-key-here")

    def get_default_model(self) -> str:
        return "gemini-3.0-flash"

    def list_models(self) -> list[str]:
        return self.MODELS.copy()
