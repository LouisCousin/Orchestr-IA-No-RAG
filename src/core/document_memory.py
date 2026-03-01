"""DocumentMemory — mémoire narrative cumulative pour la cohérence (v4.1).

Maintient un état courant (~50K tokens) qui accumule les informations
de chaque section générée pour assurer la cohérence narrative du document.

Utilisé principalement en mode CORPUS_ATLAS pour les documents extrêmes
(200-400 pages) où le corpus complet ne tient pas en contexte.
"""

import logging
from dataclasses import dataclass, field

from src.utils.token_counter import count_tokens

logger = logging.getLogger("orchestria")


@dataclass
class SectionMemoryEntry:
    """Entrée mémoire pour une section générée."""
    section_id: str
    section_title: str
    summary: str
    entities_used: list[str] = field(default_factory=list)
    facts_cited: list[str] = field(default_factory=list)
    sources_referenced: list[str] = field(default_factory=list)
    word_count: int = 0


class DocumentMemory:
    """Mémoire cumulative du document en cours de génération.

    Permet au Writer de chaque section de connaître :
    - Ce qui a déjà été écrit (résumés des sections précédentes)
    - Quelles entités ont été mentionnées (éviter les répétitions)
    - Quels faits ont été cités (éviter les contradictions)
    - Quelles sources ont été référencées (diversifier les sources)
    """

    MAX_MEMORY_TOKENS = 50_000

    def __init__(self, max_tokens: int = 50_000):
        self.MAX_MEMORY_TOKENS = max_tokens
        self._entries: list[SectionMemoryEntry] = []
        self._global_entities: set[str] = set()
        self._global_facts: list[str] = []
        self._global_sources: set[str] = set()

    def add_section(
        self,
        section_id: str,
        section_title: str,
        content: str,
        summary: str = "",
        entities: list[str] | None = None,
        facts: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> None:
        """Enregistre une section générée dans la mémoire.

        Args:
            section_id: Identifiant de la section.
            section_title: Titre de la section.
            content: Contenu complet de la section (pour compter les mots).
            summary: Résumé de la section (si vide, tronque le contenu).
            entities: Entités mentionnées dans la section.
            facts: Faits cités dans la section.
            sources: Sources référencées dans la section.
        """
        if not summary:
            summary = content[:800] + ("..." if len(content) > 800 else "")

        entry = SectionMemoryEntry(
            section_id=section_id,
            section_title=section_title,
            summary=summary,
            entities_used=entities or [],
            facts_cited=facts or [],
            sources_referenced=sources or [],
            word_count=len(content.split()),
        )

        self._entries.append(entry)
        self._global_entities.update(e.lower() for e in (entities or []))
        self._global_facts.extend(facts or [])
        self._global_sources.update(sources or [])

        # Élaguer si la mémoire dépasse le budget
        self._prune_if_needed()

    def format_for_prompt(self) -> str:
        """Formate la mémoire pour injection dans le prompt d'un Writer."""
        if not self._entries:
            return ""

        parts = ["═══ MÉMOIRE DU DOCUMENT ═══"]
        parts.append(f"Sections déjà rédigées : {len(self._entries)}")
        parts.append(f"Mots totaux : ~{sum(e.word_count for e in self._entries)}")
        parts.append("")

        # Résumés des sections
        parts.append("── Résumés des sections précédentes ──")
        for entry in self._entries:
            parts.append(f"[{entry.section_id}] {entry.section_title}")
            parts.append(f"  {entry.summary}")
            parts.append("")

        # Entités mentionnées
        if self._global_entities:
            sorted_entities = sorted(self._global_entities)
            parts.append("── Entités déjà mentionnées ──")
            parts.append(", ".join(sorted_entities[:50]))
            parts.append("")

        # Sources utilisées
        if self._global_sources:
            parts.append("── Sources déjà référencées ──")
            parts.append(", ".join(sorted(self._global_sources)))
            parts.append("")

        # Consignes de cohérence
        parts.append("── Consignes de cohérence ──")
        parts.append("- Maintiens la continuité narrative avec les sections précédentes.")
        parts.append("- Évite de répéter des informations déjà couvertes.")
        parts.append("- Référence les sections précédentes si pertinent.")
        parts.append("- Utilise la même terminologie pour les entités déjà introduites.")

        return "\n".join(parts)

    def get_entities(self) -> set[str]:
        """Retourne l'ensemble des entités mentionnées."""
        return set(self._global_entities)

    def get_sources(self) -> set[str]:
        """Retourne l'ensemble des sources référencées."""
        return set(self._global_sources)

    def get_section_count(self) -> int:
        """Retourne le nombre de sections en mémoire."""
        return len(self._entries)

    def _prune_if_needed(self) -> None:
        """Élague la mémoire si elle dépasse le budget tokens."""
        current_text = self.format_for_prompt()
        current_tokens = count_tokens(current_text)

        if current_tokens <= self.MAX_MEMORY_TOKENS:
            return

        # Stratégie : raccourcir les résumés des sections les plus anciennes
        while current_tokens > self.MAX_MEMORY_TOKENS and len(self._entries) > 1:
            oldest = self._entries[0]
            if len(oldest.summary) > 200:
                oldest.summary = oldest.summary[:200] + "..."
            else:
                # Supprimer l'entrée la plus ancienne
                self._entries.pop(0)

            current_text = self.format_for_prompt()
            current_tokens = count_tokens(current_text)

        logger.debug(
            f"[DocumentMemory] Élagage : {len(self._entries)} entrées, "
            f"~{current_tokens} tokens"
        )

    def to_dict(self) -> dict:
        """Sérialise la mémoire pour persistance."""
        return {
            "entries": [
                {
                    "section_id": e.section_id,
                    "section_title": e.section_title,
                    "summary": e.summary,
                    "entities_used": e.entities_used,
                    "facts_cited": e.facts_cited,
                    "sources_referenced": e.sources_referenced,
                    "word_count": e.word_count,
                }
                for e in self._entries
            ],
            "global_entities": sorted(self._global_entities),
            "global_sources": sorted(self._global_sources),
        }

    @classmethod
    def from_dict(cls, data: dict, max_tokens: int = 50_000) -> "DocumentMemory":
        """Reconstruit la mémoire depuis un dict sérialisé."""
        memory = cls(max_tokens=max_tokens)
        for entry_data in data.get("entries", []):
            entry = SectionMemoryEntry(
                section_id=entry_data["section_id"],
                section_title=entry_data["section_title"],
                summary=entry_data["summary"],
                entities_used=entry_data.get("entities_used", []),
                facts_cited=entry_data.get("facts_cited", []),
                sources_referenced=entry_data.get("sources_referenced", []),
                word_count=entry_data.get("word_count", 0),
            )
            memory._entries.append(entry)
        memory._global_entities = set(data.get("global_entities", []))
        memory._global_sources = set(data.get("global_sources", []))
        return memory
