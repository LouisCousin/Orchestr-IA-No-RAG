"""Système de journalisation d'Orchestr'IA."""

import logging
from datetime import datetime
from typing import Optional


class ActivityLog:
    """Journal d'activité pour le suivi en temps réel dans l'interface."""

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, message: str, level: str = "info", section: Optional[str] = None) -> None:
        """Ajoute une entrée au journal."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "section": section,
        }
        self.entries.append(entry)

    def info(self, message: str, section: Optional[str] = None) -> None:
        self.add(message, "info", section)

    def warning(self, message: str, section: Optional[str] = None) -> None:
        self.add(message, "warning", section)

    def error(self, message: str, section: Optional[str] = None) -> None:
        self.add(message, "error", section)

    def success(self, message: str, section: Optional[str] = None) -> None:
        self.add(message, "success", section)

    def get_recent(self, n: int = 50) -> list[dict]:
        """Retourne les n dernières entrées."""
        return self.entries[-n:]

    def clear(self) -> None:
        self.entries.clear()


def setup_logging() -> logging.Logger:
    """Configure le logging Python standard."""
    logger = logging.getLogger("orchestria")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
