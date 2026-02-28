"""Registre centralisé des fournisseurs IA et de leurs modèles."""

from typing import Optional


PROVIDERS_INFO = {
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "placeholder": "sk-your-openai-api-key-here",
        "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o",
    },
    "anthropic": {
        "label": "Anthropic (Claude 4.5 / Opus 4.6)",
        "env_var": "ANTHROPIC_API_KEY",
        "placeholder": "sk-ant-your-anthropic-api-key-here",
        "models": ["claude-opus-4-6", "claude-sonnet-4-5-20250514", "claude-3-5-haiku-20241022"],
        "default_model": "claude-sonnet-4-5-20250514",
    },
    "google": {
        "label": "Google Gemini 3.1",
        "env_var": "GOOGLE_API_KEY",
        "placeholder": "your-google-api-key-here",
        "models": [
            "gemini-3.1-pro-preview",
            "gemini-3.1-pro-preview-customtools",
            "gemini-3-flash-preview",
        ],
        "default_model": "gemini-3-flash-preview",
    },
}


def get_provider_info(provider_name: str) -> dict:
    """Retourne les informations d'un fournisseur par son nom."""
    if provider_name not in PROVIDERS_INFO:
        raise ValueError(
            f"Fournisseur inconnu : {provider_name!r}. "
            f"Disponibles : {list(PROVIDERS_INFO.keys())}"
        )
    return PROVIDERS_INFO[provider_name]


def get_default_model(provider_name: str) -> str:
    """Retourne le modèle par défaut d'un fournisseur.

    Raises:
        ValueError: If provider_name is not a known provider.
    """
    # B09: raise ValueError instead of returning a wrong model for unknown providers
    if provider_name not in PROVIDERS_INFO:
        raise ValueError(
            f"Fournisseur inconnu : {provider_name!r}. "
            f"Disponibles : {list(PROVIDERS_INFO.keys())}"
        )
    return PROVIDERS_INFO[provider_name]["default_model"]


def create_provider(provider_name: str, api_key: str) -> "BaseProvider":
    """Crée une instance du fournisseur sélectionné.

    Args:
        provider_name: Nom du fournisseur ("openai", "anthropic", "google").
        api_key: Clé API pour le fournisseur.

    Returns:
        Instance du provider.

    Raises:
        ValueError: If provider_name is not a known provider.
    """
    # B08: raise ValueError consistently (like get_provider_info) instead of returning None
    if provider_name == "openai":
        from src.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key)
    elif provider_name == "anthropic":
        from src.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key)
    elif provider_name == "google":
        from src.providers.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=api_key)
    raise ValueError(
        f"Fournisseur inconnu : {provider_name!r}. "
        f"Disponibles : {list(PROVIDERS_INFO.keys())}"
    )
