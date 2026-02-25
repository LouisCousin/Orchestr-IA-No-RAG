"""YAML overrides de métadonnées bibliographiques.

Phase 3 : permet à l'utilisateur de corriger manuellement les métadonnées
bibliographiques extraites par GROBID ou depuis les PDF.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.config import load_yaml, save_yaml
from src.utils.file_utils import ensure_dir

logger = logging.getLogger("orchestria")

# Champs autorisés dans un fichier override
ALLOWED_FIELDS = {
    "title", "authors", "year", "journal", "volume", "issue",
    "pages_range", "doi", "publisher", "doc_type",
}

# Plage d'années valides
MIN_YEAR = 1900
MAX_YEAR = datetime.now().year + 1


class MetadataOverrides:
    """Gère les overrides YAML de métadonnées bibliographiques."""

    def __init__(self, project_dir: Path):
        self.overrides_dir = project_dir / "corpus" / "overrides"

    def load_override(self, doc_id: str) -> dict:
        """Charge les overrides pour un document.

        Args:
            doc_id: Identifiant du document.

        Returns:
            Dict des overrides (vide si aucun).
        """
        filepath = self._get_filepath(doc_id)
        if not filepath.exists():
            return {}
        try:
            data = load_yaml(filepath)
            return self._validate(data)
        except Exception as e:
            logger.warning(f"Erreur chargement override {doc_id}: {e}")
            return {}

    def save_override(self, doc_id: str, data: dict) -> Path:
        """Sauvegarde les overrides pour un document.

        Args:
            doc_id: Identifiant du document.
            data: Dict des champs à overrider.

        Returns:
            Chemin du fichier créé/mis à jour.
        """
        validated = self._validate(data)
        filepath = self._get_filepath(doc_id)
        ensure_dir(filepath.parent)
        save_yaml(filepath, validated)
        logger.info(f"Override sauvegardé pour {doc_id}: {list(validated.keys())}")
        return filepath

    def delete_override(self, doc_id: str) -> bool:
        """Supprime les overrides d'un document.

        Returns:
            True si le fichier a été supprimé, False sinon.
        """
        filepath = self._get_filepath(doc_id)
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Override supprimé pour {doc_id}")
            return True
        return False

    def has_override(self, doc_id: str) -> bool:
        """Vérifie si un document a des overrides."""
        return self._get_filepath(doc_id).exists()

    def list_overrides(self) -> list[str]:
        """Liste les doc_id ayant des overrides."""
        if not self.overrides_dir.exists():
            return []
        return [f.stem for f in self.overrides_dir.glob("*.yaml")]

    def merge_metadata(
        self,
        doc_id: str,
        grobid_data: Optional[dict] = None,
        pdf_data: Optional[dict] = None,
    ) -> dict:
        """Fusionne les métadonnées selon l'ordre de priorité.

        Priorité : Override YAML > GROBID > PDF > Défauts.

        Args:
            doc_id: Identifiant du document.
            grobid_data: Métadonnées extraites par GROBID.
            pdf_data: Métadonnées embarquées dans le PDF.

        Returns:
            Dict des métadonnées fusionnées.
        """
        # Commencer avec les défauts
        merged = {
            "title": None,
            "authors": None,
            "year": None,
            "journal": None,
            "volume": None,
            "issue": None,
            "pages_range": None,
            "doi": None,
            "publisher": None,
            "doc_type": "unknown",
        }

        # Appliquer les métadonnées PDF (priorité basse)
        if pdf_data:
            for key, value in pdf_data.items():
                if key in merged and value:
                    merged[key] = value

        # Appliquer les métadonnées GROBID (priorité moyenne)
        if grobid_data:
            for key, value in grobid_data.items():
                if key in merged and value:
                    # Map GROBID 'pages' to 'pages_range'
                    if key == "pages":
                        merged["pages_range"] = value
                    else:
                        merged[key] = value
            # Handle pages mapping from GROBID
            if "pages" in grobid_data and grobid_data["pages"]:
                merged["pages_range"] = grobid_data["pages"]

        # Appliquer les overrides YAML (priorité maximale)
        overrides = self.load_override(doc_id)
        for key, value in overrides.items():
            if key in merged and value is not None:
                merged[key] = value

        return merged

    def _get_filepath(self, doc_id: str) -> Path:
        """Construit le chemin du fichier override."""
        # Sanitize doc_id for filesystem
        safe_id = doc_id.replace("/", "_").replace("\\", "_")
        return self.overrides_dir / f"{safe_id}.yaml"

    @staticmethod
    def _validate(data: dict) -> dict:
        """Valide un dict d'overrides.

        - Filtre les champs non autorisés (avec log warning).
        - Valide year.
        - Valide authors (doit être une liste).

        Returns:
            Dict validé (seuls les champs autorisés).
        """
        if not data:
            return {}

        validated = {}
        for key, value in data.items():
            if key not in ALLOWED_FIELDS:
                logger.warning(f"Override : champ inconnu ignoré : {key}")
                continue

            if key == "year" and value is not None:
                try:
                    year = int(value)
                    if year < MIN_YEAR or year > MAX_YEAR:
                        logger.warning(f"Override : année invalide ({year}), ignorée")
                        continue
                    validated[key] = year
                except (ValueError, TypeError):
                    logger.warning(f"Override : année invalide ({value}), ignorée")
                    continue

            elif key == "authors" and value is not None:
                if isinstance(value, list):
                    if all(isinstance(a, str) for a in value):
                        validated[key] = value
                    else:
                        logger.warning("Override : authors doit être une liste de chaînes")
                        continue
                elif isinstance(value, str):
                    # Accept single string, wrap in list
                    validated[key] = [value]
                else:
                    logger.warning(f"Override : format authors invalide ({type(value)})")
                    continue

            else:
                validated[key] = value

        return validated
