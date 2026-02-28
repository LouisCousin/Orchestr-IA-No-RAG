"""Citations et références bibliographiques APA 7e édition.

Phase 3 : pipeline de résolution des citations inline et compilation
de la bibliographie au format APA.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("orchestria")

# Types de documents supportés pour le formatage APA
DOC_TYPES = ("article", "book", "chapter", "report", "web", "thesis")


@dataclass
class CitationRef:
    """Référence de citation trouvée dans le texte."""
    raw_text: str  # Ex: "(Dupont, 2024)"
    authors: str   # Ex: "Dupont"
    year: Optional[int] = None
    resolved_doc_id: Optional[str] = None
    position: int = 0


@dataclass
class BibliographyEntry:
    """Entrée bibliographique formatée."""
    doc_id: str
    apa_reference: str
    doc_type: str = "unknown"
    metadata: dict = field(default_factory=dict)


class CitationEngine:
    """Gère les citations APA et la compilation bibliographique."""

    def __init__(self, metadata_store=None, enabled: bool = True):
        """
        Args:
            metadata_store: Instance de MetadataStore pour la résolution.
            enabled: Si False, le moteur est désactivé.
        """
        self.metadata_store = metadata_store
        self.enabled = enabled
        self._cited_doc_ids: set[str] = set()

    # ── Formatage APA ──

    @staticmethod
    def format_apa_reference(
        doc_type: str = "article",
        authors: Optional[str] = None,
        year: Optional[int] = None,
        title: Optional[str] = None,
        journal: Optional[str] = None,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages_range: Optional[str] = None,
        doi: Optional[str] = None,
        publisher: Optional[str] = None,
        url: Optional[str] = None,
        university: Optional[str] = None,
        site_name: Optional[str] = None,
        editor: Optional[str] = None,
        book_title: Optional[str] = None,
    ) -> str:
        """Formate une référence au format APA 7e édition.

        Args:
            doc_type: Type de document (article, book, chapter, report, web, thesis).
            authors: Auteurs formatés ("Dupont, J., Smith, A.").
            year: Année de publication.
            title: Titre du document.
            journal: Nom de la revue.
            volume: Numéro de volume.
            issue: Numéro de numéro.
            pages_range: Pages (ex: "123-145").
            doi: Identifiant DOI.
            publisher: Éditeur.
            url: URL.
            university: Université (pour thèses).
            site_name: Nom du site web.
            editor: Éditeur(s) de l'ouvrage collectif.
            book_title: Titre du livre (pour chapitres).

        Returns:
            Référence APA formatée.
        """
        authors_str = authors or "Auteur inconnu"
        year_str = f"({year})" if year else "(s.d.)"
        title_str = title or "Sans titre"

        if doc_type == "article":
            # Article de revue
            ref = f"{authors_str} {year_str}. {title_str}."
            if journal:
                ref += f" *{journal}*"
                if volume:
                    ref += f", *{volume}*"
                    if issue:
                        ref += f"({issue})"
                if pages_range:
                    ref += f", {pages_range}"
                ref += "."
            if doi:
                ref += f" https://doi.org/{doi}"
            return ref

        elif doc_type == "book":
            # Livre
            ref = f"{authors_str} {year_str}. *{title_str}*."
            if publisher:
                ref += f" {publisher}."
            if doi:
                ref += f" https://doi.org/{doi}"
            return ref

        elif doc_type == "chapter":
            # Chapitre de livre
            ref = f"{authors_str} {year_str}. {title_str}."
            if editor and book_title:
                ref += f" Dans {editor} (Éd.), *{book_title}*"
                if pages_range:
                    ref += f" (p. {pages_range})"
                ref += "."
            if publisher:
                ref += f" {publisher}."
            return ref

        elif doc_type == "report":
            # Rapport
            ref = f"{authors_str} {year_str}. *{title_str}*."
            if publisher:
                ref += f" {publisher}."
            if url:
                ref += f" {url}"
            return ref

        elif doc_type == "web":
            # Page web
            ref = f"{authors_str} {year_str}. {title_str}."
            if site_name:
                ref += f" *{site_name}*."
            if url:
                ref += f" {url}"
            return ref

        elif doc_type == "thesis":
            # Thèse
            ref = f"{authors_str} {year_str}. *{title_str}*"
            thesis_type = "[Thèse de doctorat"
            if university:
                thesis_type += f", {university}"
            thesis_type += "]."
            ref += f" {thesis_type}"
            if url:
                ref += f" {url}"
            return ref

        else:
            # Fallback générique
            ref = f"{authors_str} {year_str}. {title_str}."
            if publisher:
                ref += f" {publisher}."
            return ref

    @staticmethod
    def format_apa_from_metadata(doc) -> str:
        """Formate une référence APA depuis un objet DocumentMetadata.

        Args:
            doc: DocumentMetadata ou dict avec les champs requis.

        Returns:
            Référence APA formatée.
        """
        if isinstance(doc, dict):
            get = doc.get
        else:
            get = lambda k, d=None: getattr(doc, k, d)

        authors_raw = get("authors", None)
        if authors_raw and isinstance(authors_raw, str):
            try:
                authors_list = json.loads(authors_raw)
                authors = ", ".join(authors_list) if authors_list else None
            except (json.JSONDecodeError, TypeError):
                authors = authors_raw
        elif isinstance(authors_raw, list):
            authors = ", ".join(authors_raw)
        else:
            authors = None

        return CitationEngine.format_apa_reference(
            doc_type=get("doc_type", "article"),
            authors=authors,
            year=get("year", None),
            title=get("title", None),
            journal=get("journal", None),
            volume=get("volume", None),
            issue=get("issue", None),
            pages_range=get("pages_range", None),
            doi=get("doi", None),
            publisher=get("publisher", None),
            url=get("url", None),
            university=get("university", None),
            site_name=get("site_name", None),
            editor=get("editor", None),
            book_title=get("book_title", None),
        )

    # ── Parsing des citations inline ──

    @staticmethod
    def extract_inline_citations(text: str) -> list[CitationRef]:
        """Extrait les citations inline du texte généré.

        Patterns reconnus :
        - (Dupont, 2024)
        - (Dupont et al., 2024)
        - (Dupont & Smith, 2024)
        - (Dupont et Smith, 2024)

        Returns:
            Liste de CitationRef trouvées.
        """
        pattern = re.compile(
            r'\(([A-ZÀ-Ü][a-zà-ü]+(?:\s+(?:et\s+al\.|&\s+[A-ZÀ-Ü][a-zà-ü]+|et\s+[A-ZÀ-Ü][a-zà-ü]+))?),\s*(\d{4})\)'
        )
        citations = []
        for match in pattern.finditer(text):
            citations.append(CitationRef(
                raw_text=match.group(0),
                authors=match.group(1).strip(),
                year=int(match.group(2)),
                position=match.start(),
            ))
        return citations

    # ── Résolution vers les doc_id ──

    def resolve_citations(self, citations: list[CitationRef]) -> list[CitationRef]:
        """Résout les citations vers des doc_id dans le metadata_store.

        Args:
            citations: Liste de citations à résoudre.

        Returns:
            Citations avec resolved_doc_id renseigné quand possible.
        """
        if not self.metadata_store or not self.enabled:
            return citations

        all_docs = self.metadata_store.get_all_documents()

        for citation in citations:
            best_match = None
            best_score = 0

            for doc in all_docs:
                score = self._match_score(citation, doc)
                if score > best_score:
                    best_score = score
                    best_match = doc

            if best_match and best_score >= 1:
                citation.resolved_doc_id = best_match.doc_id
                self._cited_doc_ids.add(best_match.doc_id)

        return citations

    @staticmethod
    def _match_score(citation: CitationRef, doc) -> int:
        """Calcule un score de correspondance citation ↔ document."""
        score = 0
        authors_str = doc.authors or ""
        if isinstance(authors_str, str):
            try:
                authors_list = json.loads(authors_str)
            except (json.JSONDecodeError, TypeError):
                authors_list = [authors_str]
        else:
            authors_list = list(authors_str)

        # Match year
        if citation.year and doc.year and citation.year == doc.year:
            score += 1

        # Match author last name
        citation_author = citation.authors.split(" ")[0].rstrip(",").lower()
        for author in authors_list:
            if citation_author in author.lower():
                score += 2
                break

        return score

    # ── Compilation de la bibliographie ──

    def compile_bibliography(self, doc_ids: Optional[set[str]] = None) -> list[BibliographyEntry]:
        """Compile la bibliographie à partir des documents cités.

        Args:
            doc_ids: Ensemble de doc_id à inclure. Si None, utilise les doc_id
                     cités via resolve_citations().

        Returns:
            Liste triée alphabétiquement des entrées bibliographiques.
        """
        if not self.metadata_store or not self.enabled:
            return []

        ids_to_compile = doc_ids or self._cited_doc_ids
        if not ids_to_compile:
            return []

        entries = []
        for doc_id in ids_to_compile:
            doc = self.metadata_store.get_document(doc_id)
            if not doc:
                continue

            apa_ref = doc.apa_reference
            if not apa_ref:
                apa_ref = self.format_apa_from_metadata(doc)

            # Check completeness
            has_authors = bool(doc.authors)
            has_year = bool(doc.year)
            has_title = bool(doc.title)
            if not (has_authors and has_year and has_title):
                apa_ref += " [Métadonnées incomplètes]"

            entries.append(BibliographyEntry(
                doc_id=doc_id,
                apa_reference=apa_ref,
                doc_type=doc.doc_type or "unknown",
                metadata={
                    "title": doc.title,
                    "authors": doc.authors,
                    "year": doc.year,
                },
            ))

        # Sort alphabetically
        entries.sort(key=lambda e: e.apa_reference.lower())
        return entries

    def get_cited_doc_ids(self) -> set[str]:
        """Retourne l'ensemble des doc_id cités."""
        return set(self._cited_doc_ids)

    def reset(self) -> None:
        """Réinitialise les citations résolues."""
        self._cited_doc_ids.clear()
