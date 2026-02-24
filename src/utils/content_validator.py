"""Validation du contenu web acquis (détection de pages anti-bot)."""

import re
from typing import Optional


# Mots-clés indicateurs de pages anti-bot (insensible à la casse)
_ANTIBOT_KEYWORDS = [
    "anubis",
    "proof-of-work",
    "proof of work",
    "checking if you are a bot",
    "making sure you're not a bot",
    "captcha",
    "challenge-platform",
    "cloudflare",
    "just a moment",
    "verify you are human",
    "ddos protection",
    "please wait while we verify",
    "enable javascript",
    "ray id",
    "attention required",
    "checking your browser",
    "please turn javascript on",
    "access denied",
    "bot protection",
    "human verification",
]


def is_antibot_page(text: str, url: str = "") -> bool:
    """Détecte si un texte est une page de protection anti-bot.

    Args:
        text: Contenu textuel extrait de la page.
        url: URL source (optionnel, pour contexte de logging).

    Returns:
        True si le contenu semble être une page anti-bot.
    """
    if not text:
        return True

    text_lower = text.lower().strip()

    # Vérifier les mots-clés anti-bot
    keyword_matches = sum(1 for kw in _ANTIBOT_KEYWORDS if kw in text_lower)
    if keyword_matches >= 2:
        return True

    # Heuristique : texte utile très court (< 500 caractères)
    # combiné avec au moins un mot-clé
    clean_text = re.sub(r'\s+', ' ', text).strip()
    if len(clean_text) < 500 and keyword_matches >= 1:
        return True

    # Heuristique : pas de paragraphes substantiels (> 100 caractères)
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 100]
    if len(paragraphs) == 0 and len(clean_text) < 1000:
        # Très peu de contenu structuré, probablement une page de challenge
        if keyword_matches >= 1:
            return True

    return False


def is_valid_pdf_content(content: bytes) -> bool:
    """Vérifie que le contenu téléchargé est bien un PDF (magic bytes).

    Args:
        content: Contenu binaire téléchargé.

    Returns:
        True si le contenu commence par le magic byte PDF.
    """
    return content[:4] == b'%PDF'
