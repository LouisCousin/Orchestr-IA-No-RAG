"""Déduplication automatique du corpus par hash SHA-256."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.text_extractor import ExtractionResult
from src.utils.file_utils import ensure_dir, load_json, save_json

logger = logging.getLogger("orchestria")


@dataclass
class DeduplicationEntry:
    """Entrée de l'index de déduplication."""
    filename: str
    hash_binary: str
    hash_text: str
    status: str  # "unique", "doublon_exact", "doublon_probable"
    original_ref: Optional[str] = None  # Référence au fichier original si doublon


@dataclass
class DeduplicationReport:
    """Rapport de déduplication du corpus."""
    entries: list[DeduplicationEntry] = field(default_factory=list)
    exact_duplicates: int = 0
    content_duplicates: int = 0
    unique_files: int = 0
    tokens_saved: int = 0

    def to_dict(self) -> dict:
        return {
            "entries": [
                {
                    "filename": e.filename,
                    "hash_binary": e.hash_binary,
                    "hash_text": e.hash_text,
                    "status": e.status,
                    "original_ref": e.original_ref,
                }
                for e in self.entries
            ],
            "exact_duplicates": self.exact_duplicates,
            "content_duplicates": self.content_duplicates,
            "unique_files": self.unique_files,
            "tokens_saved": self.tokens_saved,
        }


class CorpusDeduplicator:
    """Détecte et gère les doublons dans le corpus."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.index_path = corpus_dir / "dedup_index.json"
        self._binary_index: dict[str, str] = {}  # hash_binary → filename
        self._text_index: dict[str, str] = {}  # hash_text → filename
        self._load_index()

    def _load_index(self) -> None:
        """Charge l'index de déduplication depuis le disque."""
        if self.index_path.exists():
            try:
                data = load_json(self.index_path)
                for entry in data.get("entries", []):
                    if entry.get("hash_binary"):
                        self._binary_index[entry["hash_binary"]] = entry["filename"]
                    if entry.get("hash_text"):
                        self._text_index[entry["hash_text"]] = entry["filename"]
            except Exception as e:
                logger.warning(f"Impossible de charger l'index de déduplication : {e}")

    def save_index(self, report: DeduplicationReport) -> None:
        """Persiste l'index de déduplication sur le disque.

        Fusionne les entrées du rapport avec les entrées précédemment
        enregistrées dans l'index afin de ne pas perdre l'historique.
        """
        # Charger les entrées existantes depuis le fichier pour les conserver
        existing_entries = {}
        if self.index_path.exists():
            try:
                data = load_json(self.index_path)
                for entry in data.get("entries", []):
                    existing_entries[entry["filename"]] = entry
            except Exception:
                pass

        # Fusionner : les nouvelles entrées remplacent les anciennes pour le même filename
        for entry in report.to_dict()["entries"]:
            existing_entries[entry["filename"]] = entry

        merged_data = {
            "entries": list(existing_entries.values()),
            "exact_duplicates": report.exact_duplicates,
            "content_duplicates": report.content_duplicates,
            "unique_files": report.unique_files,
            "tokens_saved": report.tokens_saved,
        }
        save_json(self.index_path, merged_data)

    def check_duplicate(self, extraction: ExtractionResult) -> DeduplicationEntry:
        """Vérifie si un document est un doublon.

        Retourne une DeduplicationEntry avec le statut approprié.
        """
        filename = extraction.source_filename

        # Vérification par hash binaire (doublons exacts)
        if extraction.hash_binary and extraction.hash_binary in self._binary_index:
            original = self._binary_index[extraction.hash_binary]
            logger.info(f"Doublon exact détecté : {filename} ↔ {original}")
            return DeduplicationEntry(
                filename=filename,
                hash_binary=extraction.hash_binary,
                hash_text=extraction.hash_text,
                status="doublon_exact",
                original_ref=original,
            )

        # Vérification par hash textuel (doublons de contenu)
        if extraction.hash_text and extraction.hash_text in self._text_index:
            original = self._text_index[extraction.hash_text]
            logger.info(f"Doublon de contenu détecté : {filename} ↔ {original}")
            return DeduplicationEntry(
                filename=filename,
                hash_binary=extraction.hash_binary,
                hash_text=extraction.hash_text,
                status="doublon_probable",
                original_ref=original,
            )

        # Document unique
        return DeduplicationEntry(
            filename=filename,
            hash_binary=extraction.hash_binary,
            hash_text=extraction.hash_text,
            status="unique",
        )

    def register(self, entry: DeduplicationEntry) -> None:
        """Enregistre un document dans l'index."""
        if entry.hash_binary:
            self._binary_index[entry.hash_binary] = entry.filename
        if entry.hash_text:
            self._text_index[entry.hash_text] = entry.filename

    def deduplicate_corpus(
        self, extractions: list[ExtractionResult]
    ) -> DeduplicationReport:
        """Analyse une liste d'extractions et produit un rapport de déduplication."""
        report = DeduplicationReport()

        for extraction in extractions:
            entry = self.check_duplicate(extraction)
            report.entries.append(entry)

            if entry.status == "doublon_exact":
                report.exact_duplicates += 1
                report.tokens_saved += len(extraction.text) // 4  # Estimation
            elif entry.status == "doublon_probable":
                report.content_duplicates += 1
                report.tokens_saved += len(extraction.text) // 4  # Estimation
            else:
                report.unique_files += 1
                self.register(entry)

        self.save_index(report)
        return report

    def remove_duplicate(self, filename: str) -> bool:
        """Supprime un fichier doublon du corpus."""
        file_path = self.corpus_dir / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Doublon supprimé : {filename}")
            return True
        return False
