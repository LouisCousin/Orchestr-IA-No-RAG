"""Fournisseur Google Gemini pour Orchestr'IA — Phase 5.

Migration vers Gemini 3.1 avec support du context caching explicite
et du paramètre thinking_level.
"""

import logging
import os
import time
from typing import Optional

from src.providers.base import AIResponse, BaseProvider

logger = logging.getLogger("orchestria")

try:
    from google.genai import types
except ImportError:
    types = None  # type: ignore

# Modèles compatibles avec le context caching explicite
_CACHING_COMPATIBLE_MODELS = {
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
}

# Famille de modèles supportant thinking_level
_THINKING_LEVEL_MODELS = {
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
}


class GeminiProvider(BaseProvider):
    """Fournisseur Google Gemini 3.1 avec context caching et thinking_level."""

    MODELS = [
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
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
        cached_content: Optional[str] = None,
        thinking_level: Optional[str] = None,
    ) -> AIResponse:
        """Génère du contenu via l'API Google Gemini avec retry automatique.

        Args:
            prompt: Texte de l'invite utilisateur.
            system_prompt: Instruction système. Ignorée si cached_content est fourni
                           (le system_prompt est déjà dans le cache).
            model: Modèle à utiliser. Défaut : gemini-3-flash-preview.
            temperature: Créativité de la génération (0=déterministe, 1=créatif).
            max_tokens: Nombre maximum de tokens de sortie.
            cached_content: Nom du cache Gemini (ex: "cachedContents/abc123").
                            Si fourni, system_prompt est ignoré.
            thinking_level: Profondeur de raisonnement interne. Valeurs acceptées :
                            "minimal", "low", "medium", "high". Ignoré pour Flash.
        """
        model = model or self.get_default_model()

        if types is None:
            raise ImportError(
                "Le package google-genai n'est pas installé. "
                "Installez-le avec : pip install google-genai"
            )

        # R2 : Avertir si cached_content ET system_prompt sont fournis
        if cached_content and system_prompt:
            logger.warning(
                "cached_content et system_prompt fournis simultanément. "
                "system_prompt sera ignoré car il est déjà inclus dans le cache."
            )

        last_error = None
        cache_retried = False
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()

                # Construire la config de génération
                config_kwargs = dict(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                # R1 : si cached_content fourni, ne pas inclure system_instruction
                if cached_content:
                    config_kwargs["cached_content"] = cached_content
                elif system_prompt:
                    config_kwargs["system_instruction"] = system_prompt

                # R3 : thinking_level uniquement pour les modèles 3.1
                if thinking_level and model in _THINKING_LEVEL_MODELS:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=self._thinking_level_to_budget(thinking_level)
                    )
                elif thinking_level and model not in _THINKING_LEVEL_MODELS:
                    logger.debug(
                        f"thinking_level ignoré pour {model} (non compatible famille 3.1)."
                    )

                config = types.GenerateContentConfig(**config_kwargs)

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                # response.text peut lever ValueError si la réponse est bloquée
                try:
                    content = response.text or ""
                except ValueError:
                    content = ""
                    logger.warning("Réponse Gemini bloquée par les filtres de sécurité")

                # Extraire les tokens depuis usage_metadata
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
                if response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    cached_tokens = getattr(response.usage_metadata, "cached_content_token_count", 0) or 0

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
                    raw_response={
                        "cached_tokens": cached_tokens,
                        "cache_name": cached_content,
                    },
                )

            except Exception as e:
                error_str = str(e)
                last_error = e

                # Gestion erreurs liées au cache — ne pas consommer le budget de retry
                is_cache_error = False
                if "cached_content and tools are mutually exclusive" in error_str:
                    logger.error(
                        "Conflit cached_content/tools détecté (erreur 400). "
                        "Relance sans cache."
                    )
                    is_cache_error = True
                elif "minimum token count" in error_str.lower():
                    logger.error(
                        "Corpus trop petit pour le caching (erreur 400). "
                        "Basculement en mode standard."
                    )
                    is_cache_error = True
                elif "cachedContent" in error_str and "not found" in error_str.lower():
                    logger.error(
                        f"Cache Gemini introuvable ou expiré : {cached_content}. "
                        "Basculement en mode standard."
                    )
                    is_cache_error = True

                if is_cache_error and not cache_retried:
                    cached_content = None
                    cache_retried = True
                    continue

                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        f"Erreur API Gemini (tentative {attempt + 1}/{self._max_retries + 1}): {e}. "
                        f"Retry dans {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(f"Échec après {self._max_retries + 1} tentatives: {last_error}")

    def _thinking_level_to_budget(self, thinking_level: str) -> int:
        """Convertit un niveau textuel en budget de tokens de réflexion."""
        mapping = {
            "minimal": 128,
            "low": 1024,
            "medium": 8192,
            "high": 32768,
        }
        return mapping.get(thinking_level, 8192)

    def supports_caching(self, model: Optional[str] = None) -> bool:
        """Retourne True si le modèle supporte le context caching explicite."""
        m = model or self.get_default_model()
        return m in _CACHING_COMPATIBLE_MODELS

    def supports_batch(self) -> bool:
        """Gemini ne supporte pas le batch API dans cette version."""
        return False

    def is_available(self) -> bool:
        """Vérifie si la clé API est configurée."""
        return bool(self._api_key and self._api_key != "your-google-api-key-here")

    def get_default_model(self) -> str:
        return "gemini-3-flash-preview"

    def list_models(self) -> list[str]:
        return self.MODELS.copy()
