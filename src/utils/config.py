"""Gestion de la configuration globale d'Orchestr'IA."""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"


def load_env() -> None:
    """Charge les variables d'environnement depuis .env."""
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def load_yaml(path: Path) -> dict:
    """Charge un fichier YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    """Sauvegarde un dictionnaire en YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_default_config() -> dict:
    """Charge la configuration par défaut."""
    return load_yaml(CONFIG_DIR / "default.yaml")


def load_model_pricing() -> dict:
    """Charge les tarifs des modèles."""
    return load_yaml(CONFIG_DIR / "model_pricing.yaml")


def get_nested(data: dict, key_path: str, default: Any = None) -> Any:
    """Accède à une valeur imbriquée via un chemin pointé (ex: 'generation.temperature')."""
    keys = key_path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
