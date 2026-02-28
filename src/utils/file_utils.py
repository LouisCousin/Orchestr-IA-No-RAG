"""Utilitaires de gestion de fichiers."""

import hashlib
import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Optional

# B16: thread-safe lock for sequence number allocation (TOCTOU fix)
_seq_lock = threading.Lock()


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
    """Charge un fichier JSON (attend un objet à la racine)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


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
    """Nettoie un nom de fichier pour le rendre sûr.

    Protège contre le path traversal et les caractères spéciaux.
    """
    # Sécurité contre path traversal : ne garder que le nom de base
    name = Path(name).name
    # Garder uniquement alphanumérique, tirets, underscores et points
    name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', name)
    name = name.strip("._")
    return name[:100] if name else "unnamed"


def get_next_sequence_number(directory: Path) -> int:
    """Retourne le prochain numéro séquentiel pour les fichiers du corpus.

    B16: Uses a threading lock to prevent TOCTOU race conditions, and
    supports sequence numbers >= 1000 via relaxed regex.
    """
    with _seq_lock:
        existing = list(directory.glob("*"))
        max_num = 0
        for f in existing:
            # B16: changed from r"^(\d{3})_" to r"^(\d+)_" to support >= 1000
            match = re.match(r"^(\d+)_", f.name)
            if match:
                max_num = max(max_num, int(match.group(1)))
        return max_num + 1


def format_sequence_name(seq_num: int, source_name: str, ext: str) -> str:
    """Formate un nom de fichier séquentiel : NNN_source.ext (or NNNN for seq >= 1000)."""
    clean_name = sanitize_filename(source_name)
    # B16: use at least 3 digits, expanding to 4+ for seq >= 1000
    width = max(3, len(str(seq_num)))
    return f"{seq_num:0{width}d}_{clean_name}{ext}"


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
