"""Nettoyage des références [Source N] résiduelles dans le texte généré."""

import re
from typing import Optional


_SOURCE_PATTERN = re.compile(r'\[Source\s*\d+\]')


def clean_source_references(text: str, source_map: Optional[dict] = None) -> str:
    """Supprime ou remplace les références [Source N] dans le texte.

    Args:
        text: Texte contenant potentiellement des [Source N].
        source_map: Dict optionnel {numéro (int) -> nom du fichier source}.
            Si fourni, remplace [Source N] par le nom du fichier.
            Sinon, supprime simplement les occurrences.

    Returns:
        Texte nettoyé sans [Source N].
    """
    if source_map:
        def _replace(match):
            num_match = re.search(r'\d+', match.group(0))
            if num_match:
                num = int(num_match.group())
                if num in source_map:
                    return source_map[num]
            return ""
        text = _SOURCE_PATTERN.sub(_replace, text)
    else:
        text = _SOURCE_PATTERN.sub("", text)

    # Nettoyer les espaces doubles résultants
    text = re.sub(r'  +', ' ', text)
    return text


def has_source_references(text: str) -> bool:
    """Vérifie si le texte contient des références [Source N]."""
    return bool(_SOURCE_PATTERN.search(text))
