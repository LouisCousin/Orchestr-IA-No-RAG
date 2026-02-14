"""Utilitaires de gestion de fichiers."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional


def ensure_dir(path: Path) -> Path:
    """Crée un répertoire s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    """Lit un fichier texte avec fallback d'encodage."""
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def save_json(path: Path, data: dict) -> None:
    """Sauvegarde un dictionnaire en JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    """Charge un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    """Calcule le hash SHA-256 du contenu binaire d'un fichier."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_text(text: str) -> str:
    """Calcule le hash SHA-256 d'un texte normalisé."""
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_text_for_hash(text: str) -> str:
    """Normalise un texte pour la comparaison de hash (minuscules, espaces réduits, pas de ponctuation)."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_filename(name: str) -> str:
    """Nettoie un nom pour l'utiliser comme nom de fichier."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name[:100] if name else "unnamed"


def get_next_sequence_number(directory: Path) -> int:
    """Retourne le prochain numéro séquentiel pour les fichiers du corpus."""
    existing = list(directory.glob("*"))
    max_num = 0
    for f in existing:
        match = re.match(r"^(\d{3})_", f.name)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def format_sequence_name(seq_num: int, source_name: str, ext: str) -> str:
    """Formate un nom de fichier séquentiel : NNN_source.ext."""
    clean_name = sanitize_filename(source_name)
    return f"{seq_num:03d}_{clean_name}{ext}"


def get_mime_type(path: Path) -> Optional[str]:
    """Détermine le type MIME basique d'un fichier."""
    ext = path.suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
    }
    return mime_map.get(ext)
