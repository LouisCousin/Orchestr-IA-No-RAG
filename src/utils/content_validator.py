"""Validation du contenu web acquis (détection de pages anti-bot)."""

import re


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

    La détection nécessite TOUJOURS au moins 1 mot-clé anti-bot.
    La longueur seule n'est pas un signal suffisant (évite les faux
    positifs sur les pages légitimes courtes).

    Args:
        text: Contenu textuel extrait de la page.
        url: URL source (optionnel, pour contexte de logging).

    Returns:
        True si le contenu semble être une page anti-bot.
    """
    if not text or not text.strip():
        return False

    text_lower = text.lower().strip()

    # Vérifier les mots-clés anti-bot
    keyword_matches = sum(1 for kw in _ANTIBOT_KEYWORDS if kw in text_lower)

    # Critère 1 : 2+ mots-clés = anti-bot certain
    if keyword_matches >= 2:
        return True

    # Critère 2 : 1 mot-clé + contenu très court = suspect
    clean_text = re.sub(r'\s+', ' ', text).strip()
    if keyword_matches >= 1 and len(clean_text) < 500:
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
