"""Gestion des profils de projet préconfigurés."""

import logging
from pathlib import Path
from typing import Optional

from src.utils.config import load_yaml, save_yaml, ROOT_DIR

logger = logging.getLogger("orchestria")

PROFILES_DIR = ROOT_DIR / "profiles" / "default"


# 5 profils préconfigurés livrés avec l'application
DEFAULT_PROFILES = {
    "rapport_analyse": {
        "name": "Rapport d'analyse",
        "description": "Rapport structuré d'analyse approfondie d'un sujet, avec données, constats et recommandations.",
        "target_pages": 20,
        "tone": "professionnel et analytique",
        "generation": {
            "temperature": 0.6,
            "max_tokens": 4096,
            "number_of_passes": 1,
        },
        "checkpoints": {
            "after_plan_validation": True,
            "after_corpus_acquisition": False,
            "after_extraction": False,
            "after_prompt_generation": False,
            "after_generation": False,
            "final_review": True,
        },
        "styling": {
            "primary_color": "#2C3E50",
            "secondary_color": "#34495E",
            "font_title": "Calibri",
            "font_body": "Calibri",
            "font_size_title": 16,
            "font_size_body": 11,
        },
        "persistent_instructions": "Adopte un ton analytique et factuel. Chaque affirmation doit être étayée par des données du corpus. Inclus des tableaux et des listes à puces pour structurer les informations clés.",
    },
    "proposition_services": {
        "name": "Proposition de services",
        "description": "Document commercial de proposition de services ou de réponse à appel d'offres.",
        "target_pages": 15,
        "tone": "professionnel et persuasif",
        "generation": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "number_of_passes": 1,
        },
        "checkpoints": {
            "after_plan_validation": True,
            "after_corpus_acquisition": False,
            "after_extraction": False,
            "after_prompt_generation": True,
            "after_generation": False,
            "final_review": True,
        },
        "styling": {
            "primary_color": "#F0C441",
            "secondary_color": "#4E4E50",
            "font_title": "Calibri",
            "font_body": "Calibri",
            "font_size_title": 16,
            "font_size_body": 11,
        },
        "persistent_instructions": "Adopte un ton professionnel et orienté bénéfices client. Met en avant la valeur ajoutée et les avantages concurrentiels. Utilise un vocabulaire positif et orienté solutions.",
    },
    "synthese_veille": {
        "name": "Synthèse de veille",
        "description": "Document de synthèse d'une veille thématique (technologique, réglementaire, concurrentielle).",
        "target_pages": 10,
        "tone": "informatif et synthétique",
        "generation": {
            "temperature": 0.5,
            "max_tokens": 4096,
            "number_of_passes": 1,
        },
        "checkpoints": {
            "after_plan_validation": True,
            "after_corpus_acquisition": False,
            "after_extraction": False,
            "after_prompt_generation": False,
            "after_generation": False,
            "final_review": True,
        },
        "styling": {
            "primary_color": "#1ABC9C",
            "secondary_color": "#2C3E50",
            "font_title": "Calibri",
            "font_body": "Calibri",
            "font_size_title": 16,
            "font_size_body": 11,
        },
        "persistent_instructions": "Adopte un ton informatif et concis. Privilégie la synthèse à l'exhaustivité. Identifie les tendances clés et les signaux faibles. Chaque section doit commencer par un point clé encadré.",
    },
    "document_formation": {
        "name": "Document de formation",
        "description": "Support pédagogique ou guide de formation structuré par modules.",
        "target_pages": 25,
        "tone": "pédagogique et accessible",
        "generation": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "number_of_passes": 1,
        },
        "checkpoints": {
            "after_plan_validation": True,
            "after_corpus_acquisition": False,
            "after_extraction": False,
            "after_prompt_generation": False,
            "after_generation": True,
            "final_review": True,
        },
        "styling": {
            "primary_color": "#3498DB",
            "secondary_color": "#2C3E50",
            "font_title": "Calibri",
            "font_body": "Calibri",
            "font_size_title": 16,
            "font_size_body": 11,
        },
        "persistent_instructions": "Adopte un ton pédagogique et progressif. Chaque concept doit être expliqué avant d'être utilisé. Inclus des exemples concrets, des définitions encadrées et des points de synthèse à la fin de chaque section.",
    },
    "compte_rendu": {
        "name": "Compte rendu",
        "description": "Compte rendu de réunion, d'événement ou de mission, structuré et factuel.",
        "target_pages": 5,
        "tone": "factuel et concis",
        "generation": {
            "temperature": 0.4,
            "max_tokens": 4096,
            "number_of_passes": 1,
        },
        "checkpoints": {
            "after_plan_validation": True,
            "after_corpus_acquisition": False,
            "after_extraction": False,
            "after_prompt_generation": False,
            "after_generation": False,
            "final_review": True,
        },
        "styling": {
            "primary_color": "#7F8C8D",
            "secondary_color": "#2C3E50",
            "font_title": "Calibri",
            "font_body": "Calibri",
            "font_size_title": 14,
            "font_size_body": 11,
        },
        "persistent_instructions": "Adopte un ton factuel et neutre. Privilégie les phrases courtes et les listes à puces. Chaque point doit être attribuable (qui a dit/fait quoi). Inclus systématiquement les décisions prises et les actions à mener.",
    },
}


class ProfileManager:
    """Gère les profils de projet."""

    def __init__(self):
        self.profiles_dir = PROFILES_DIR
        self._ensure_default_profiles()

    def _ensure_default_profiles(self) -> None:
        """Crée les profils par défaut s'ils n'existent pas."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        for profile_id, profile_data in DEFAULT_PROFILES.items():
            profile_path = self.profiles_dir / f"{profile_id}.yaml"
            if not profile_path.exists():
                save_yaml(profile_path, profile_data)
                logger.info(f"Profil par défaut créé : {profile_data['name']}")

    def list_profiles(self) -> list[dict]:
        """Liste tous les profils disponibles."""
        profiles = []
        for yaml_file in sorted(self.profiles_dir.glob("*.yaml")):
            try:
                data = load_yaml(yaml_file)
                profiles.append({
                    "id": yaml_file.stem,
                    "name": data.get("name", yaml_file.stem),
                    "description": data.get("description", ""),
                    "target_pages": data.get("target_pages"),
                    "path": str(yaml_file),
                })
            except Exception as e:
                logger.warning(f"Erreur chargement profil {yaml_file}: {e}")
        return profiles

    def load_profile(self, profile_id: str) -> Optional[dict]:
        """Charge un profil par son identifiant."""
        profile_path = self.profiles_dir / f"{profile_id}.yaml"
        if not profile_path.exists():
            # Chercher dans les profils personnalisés
            custom_path = ROOT_DIR / "profiles" / "custom" / f"{profile_id}.yaml"
            if custom_path.exists():
                profile_path = custom_path
            else:
                logger.warning(f"Profil non trouvé : {profile_id}")
                return None

        try:
            return load_yaml(profile_path)
        except Exception as e:
            logger.error(f"Erreur chargement profil {profile_id}: {e}")
            return None

    def get_profile_config(self, profile_id: str) -> dict:
        """Retourne la configuration complète d'un profil, prête à être utilisée."""
        profile = self.load_profile(profile_id)
        if not profile:
            return {}

        return {
            "model": profile.get("model", "gpt-4o"),
            "temperature": profile.get("generation", {}).get("temperature", 0.7),
            "max_tokens": profile.get("generation", {}).get("max_tokens", 4096),
            "target_pages": profile.get("target_pages"),
            "tone": profile.get("tone", ""),
            "persistent_instructions": profile.get("persistent_instructions", ""),
            "checkpoints": profile.get("checkpoints", {}),
            "styling": profile.get("styling", {}),
            "number_of_passes": profile.get("generation", {}).get("number_of_passes", 1),
        }
