"""Chunking sémantique hiérarchique pour le pipeline RAG.

Phase 2.5 : remplace le chunking fixe par un découpage intelligent
qui respecte les frontières de sections, paragraphes et phrases.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.core.text_extractor import ExtractionResult

logger = logging.getLogger("orchestria")

# Paramètres par défaut (surchargés par config/default.yaml)
DEFAULT_MAX_CHUNK_TOKENS = 800
DEFAULT_MIN_CHUNK_TOKENS = 100
DEFAULT_OVERLAP_SENTENCES = 2


@dataclass
class Chunk:
    """Bloc de texte sémantique avec métadonnées."""
    doc_id: str           # Lien vers le document source (UUID)
    text: str             # Contenu textuel du chunk
    page_number: int      # Page source dans le PDF
    section_title: str    # Titre de la section d'origine
    chunk_index: int      # Position dans le document (0, 1, 2, ...)
    chunk_id: str = ""    # Généré automatiquement : f"{doc_id}_{chunk_index}"
    token_count: int = 0  # Calculé après création

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}_{self.chunk_index:04d}"
        if not self.token_count:
            self.token_count = _count_tokens(self.text)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
        }


def _count_tokens(text: str) -> int:
    """Estimation du nombre de tokens (heuristique : ~4 chars/token en français)."""
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases."""
    import re
    # Découpe sur ". ", "! ", "? " suivis d'une majuscule ou fin de texte
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _split_into_sentence_groups(text: str, max_chunk_tokens: int) -> list[str]:
    """Découpe un texte long en groupes de phrases respectant la limite de tokens."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        # Dernier recours : découpage brut par caractères
        max_chars = max_chunk_tokens * 4  # consistent with _count_tokens: ~4 chars/token
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    groups = []
    current = ""
    for sentence in sentences:
        candidate = current + " " + sentence if current else sentence
        if _count_tokens(candidate) > max_chunk_tokens and current:
            groups.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        groups.append(current.strip())
    return groups


def chunk_document(
    extraction: ExtractionResult,
    doc_id: str,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Chunking sémantique hiérarchique.

    Priorité 1 : découpage par sections détectées (Docling/python-docx).
    Priorité 2 : sections longues découpées par paragraphes.
    Priorité 3 : fallback par tokens (texte brut sans structure).
    """
    if extraction.structure:
        return _chunk_by_sections(
            extraction.structure, doc_id,
            max_chunk_tokens, min_chunk_tokens, overlap_sentences,
        )
    else:
        return _chunk_by_tokens(
            extraction.text, doc_id,
            max_chunk_tokens, min_chunk_tokens, overlap_sentences,
        )


def _chunk_by_sections(
    sections: list[dict],
    doc_id: str,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Découpe par sections sémantiques (Priorités 1 et 2)."""
    chunks: list[Chunk] = []
    current_section_title = "Introduction"
    current_page = 1

    for section in sections:
        section_type = section.get("type", "paragraph")
        section_text = section.get("text", "")

        if not section_text.strip():
            continue

        if section_type == "title" or section.get("level", 0) > 0:
            current_section_title = section_text
            current_page = section.get("page") or current_page
            continue

        text = section_text
        page = section.get("page") or current_page
        token_count = _count_tokens(text)

        if token_count <= max_chunk_tokens:
            # Section courte → 1 chunk
            if token_count < min_chunk_tokens and chunks and chunks[-1].section_title == current_section_title:
                # Trop court → fusionner avec le chunk précédent (même section uniquement)
                # B20: only merge if the result won't exceed max_chunk_tokens
                merged_count = _count_tokens(chunks[-1].text + "\n" + text)
                if merged_count <= max_chunk_tokens:
                    chunks[-1].text += "\n" + text
                    chunks[-1].token_count = merged_count
                else:
                    # Would exceed limit: keep as separate chunk
                    chunks.append(Chunk(
                        doc_id=doc_id,
                        text=text,
                        page_number=page,
                        section_title=current_section_title,
                        chunk_index=len(chunks),
                    ))
            else:
                chunks.append(Chunk(
                    doc_id=doc_id,
                    text=text,
                    page_number=page,
                    section_title=current_section_title,
                    chunk_index=len(chunks),
                ))
        else:
            # ═══ PRIORITÉ 2 : Section longue → découpage par paragraphes ═══
            paragraphs = text.split("\n\n")
            # Si un seul paragraphe dépasse max_chunk_tokens, découper par phrases
            if len(paragraphs) <= 1 and _count_tokens(text) > max_chunk_tokens:
                paragraphs = _split_into_sentence_groups(text, max_chunk_tokens)

            buffer = ""
            for para in paragraphs:
                candidate = buffer + "\n" + para if buffer else para
                if _count_tokens(candidate) > max_chunk_tokens and buffer:
                    chunks.append(Chunk(
                        doc_id=doc_id,
                        text=buffer.strip(),
                        page_number=page,
                        section_title=current_section_title,
                        chunk_index=len(chunks),
                    ))
                    # Overlap : reprendre les dernières phrases
                    sentences = _split_sentences(buffer)
                    overlap_text = ". ".join(sentences[-overlap_sentences:]) if len(sentences) >= overlap_sentences else ""
                    buffer = overlap_text + "\n" + para if overlap_text else para
                else:
                    buffer = candidate

            if buffer.strip():
                buf_tokens = _count_tokens(buffer)
                if buf_tokens < min_chunk_tokens and chunks and chunks[-1].section_title == current_section_title:
                    # B20: only merge if the result won't exceed max_chunk_tokens
                    merged_count = _count_tokens(chunks[-1].text + "\n" + buffer.strip())
                    if merged_count <= max_chunk_tokens:
                        chunks[-1].text += "\n" + buffer.strip()
                        chunks[-1].token_count = merged_count
                    else:
                        chunks.append(Chunk(
                            doc_id=doc_id,
                            text=buffer.strip(),
                            page_number=page,
                            section_title=current_section_title,
                            chunk_index=len(chunks),
                        ))
                else:
                    chunks.append(Chunk(
                        doc_id=doc_id,
                        text=buffer.strip(),
                        page_number=page,
                        section_title=current_section_title,
                        chunk_index=len(chunks),
                    ))

    # Reindex after potential merges
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.chunk_id = f"{doc_id}_{i:04d}"

    return chunks


def _chunk_by_tokens(
    text: str,
    doc_id: str,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Fallback : découpage par tokens quand aucune structure n'est disponible (Priorité 3)."""
    if not text or not text.strip():
        return []

    max_chars = max_chunk_tokens * 4  # consistent with _count_tokens: ~4 chars/token
    chunks: list[Chunk] = []
    paragraphs = text.split("\n\n")
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = buffer + "\n\n" + para if buffer else para
        if len(candidate) > max_chars and buffer:
            chunks.append(Chunk(
                doc_id=doc_id,
                text=buffer.strip(),
                page_number=1,
                section_title="",
                chunk_index=len(chunks),
            ))
            # Overlap
            sentences = _split_sentences(buffer)
            overlap_text = ". ".join(sentences[-overlap_sentences:]) if len(sentences) >= overlap_sentences else ""
            buffer = overlap_text + "\n\n" + para if overlap_text else para
        else:
            buffer = candidate

    if buffer.strip():
        token_count = _count_tokens(buffer)
        if token_count < min_chunk_tokens and chunks:
            chunks[-1].text += "\n\n" + buffer.strip()
            chunks[-1].token_count = _count_tokens(chunks[-1].text)
        else:
            chunks.append(Chunk(
                doc_id=doc_id,
                text=buffer.strip(),
                page_number=1,
                section_title="",
                chunk_index=len(chunks),
            ))

    # Reindex
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.chunk_id = f"{doc_id}_{i:04d}"

    return chunks
