"""Registre centralisé des fournisseurs IA et de leurs modèles."""

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
        "models": ["claude-opus-4-6", "claude-sonnet-4-5-20250514", "claude-haiku-35-20241022"],
        "default_model": "claude-sonnet-4-5-20250514",
    },
    "google": {
        "label": "Google Gemini 3",
        "env_var": "GOOGLE_API_KEY",
        "placeholder": "your-google-api-key-here",
        "models": ["gemini-3.0-pro", "gemini-3.0-flash"],
        "default_model": "gemini-3.0-flash",
    },
}


def get_provider_info(provider_name: str) -> dict:
    """Retourne les informations d'un fournisseur par son nom."""
    return PROVIDERS_INFO.get(provider_name, PROVIDERS_INFO["openai"])


def get_default_model(provider_name: str) -> str:
    """Retourne le modèle par défaut d'un fournisseur."""
    info = PROVIDERS_INFO.get(provider_name, {})
    return info.get("default_model", "gpt-4o")
