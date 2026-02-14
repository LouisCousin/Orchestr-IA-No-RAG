"""Structuration du corpus extrait pour la génération.

Phase 1 : découpage simple par page/section.
Phase 2 : RAG avec ChromaDB.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.text_extractor import ExtractionResult, extract

logger = logging.getLogger("orchestria")


@dataclass
class CorpusChunk:
    """Bloc de texte du corpus."""
    text: str
    source_file: str
    chunk_index: int
    char_count: int = 0
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class StructuredCorpus:
    """Corpus structuré prêt pour la génération."""
    chunks: list[CorpusChunk] = field(default_factory=list)
    total_tokens: int = 0
    total_chunks: int = 0
    source_files: list[str] = field(default_factory=list)
    extractions: list[ExtractionResult] = field(default_factory=list)

    def get_chunks_for_section(self, section_title: str, max_chunks: int = 10) -> list[CorpusChunk]:
        """Retourne les chunks les plus pertinents pour une section (Phase 1 : tous les chunks)."""
        return self.chunks[:max_chunks]

    def get_corpus_digest(self, max_total_chars: int = 8000) -> dict:
        """Retourne un digest représentatif du corpus, adapté à sa taille.

        Trois paliers selon le nombre de documents :
        - **full_excerpts** (1-10 docs) : extrait long du début de chaque doc,
          budget réparti équitablement.
        - **first_sentences** (11-50 docs) : nom du fichier + première phrase
          de chaque document.
        - **sampled** (51+ docs) : liste de tous les noms de fichiers + extraits
          longs pour un échantillon régulier de ~5 documents.

        Args:
            max_total_chars: Budget total en caractères (~2000 tokens).

        Returns:
            Dict avec les clés :
            - ``tier`` : ``"full_excerpts"`` | ``"first_sentences"`` | ``"sampled"``
            - ``num_documents`` : nombre total de documents
            - ``entries`` : liste de dicts ``{"source_file", "text"}``
            - ``all_filenames`` : (palier *sampled* uniquement) liste de tous
              les noms de fichiers
        """
        if not self.chunks:
            return {"tier": "full_excerpts", "num_documents": 0, "entries": []}

        chunks_by_file = self._group_chunks_by_file()
        num_files = len(chunks_by_file)
        sorted_files = sorted(chunks_by_file.keys())

        if num_files <= 10:
            return self._digest_full_excerpts(chunks_by_file, sorted_files, max_total_chars)
        elif num_files <= 50:
            return self._digest_first_sentences(chunks_by_file, sorted_files)
        else:
            return self._digest_sampled(chunks_by_file, sorted_files, max_total_chars)

    # -- Helpers internes pour get_corpus_digest --

    def _group_chunks_by_file(self) -> dict[str, list["CorpusChunk"]]:
        """Regroupe les chunks par fichier source, triés par index."""
        chunks_by_file: dict[str, list[CorpusChunk]] = {}
        for chunk in self.chunks:
            chunks_by_file.setdefault(chunk.source_file, []).append(chunk)
        for file_chunks in chunks_by_file.values():
            file_chunks.sort(key=lambda c: c.chunk_index)
        return chunks_by_file

    def _digest_full_excerpts(
        self,
        chunks_by_file: dict[str, list["CorpusChunk"]],
        sorted_files: list[str],
        max_total_chars: int,
    ) -> dict:
        """Palier 1-10 docs : extrait long du début de chaque document."""
        budget_per_file = max_total_chars // len(sorted_files)
        entries = []
        for source_file in sorted_files:
            excerpt = ""
            for chunk in chunks_by_file[source_file]:
                remaining = budget_per_file - len(excerpt)
                if remaining <= 0:
                    break
                excerpt += chunk.text[:remaining]
            if excerpt:
                entries.append({"source_file": source_file, "text": excerpt.strip()})
        return {
            "tier": "full_excerpts",
            "num_documents": len(sorted_files),
            "entries": entries,
        }

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        """Extrait la première phrase non-vide d'un texte."""
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Couper à la première fin de phrase ou limiter à 200 chars
            for sep in (". ", ".\n", ".\t"):
                pos = line.find(sep)
                if 0 < pos < 200:
                    return line[: pos + 1]
            return line[:200]
        return text[:200]

    def _digest_first_sentences(
        self,
        chunks_by_file: dict[str, list["CorpusChunk"]],
        sorted_files: list[str],
    ) -> dict:
        """Palier 11-50 docs : nom du fichier + première phrase."""
        entries = []
        for source_file in sorted_files:
            first_chunk = chunks_by_file[source_file][0]
            sentence = self._extract_first_sentence(first_chunk.text)
            entries.append({"source_file": source_file, "text": sentence})
        return {
            "tier": "first_sentences",
            "num_documents": len(sorted_files),
            "entries": entries,
        }

    def _digest_sampled(
        self,
        chunks_by_file: dict[str, list["CorpusChunk"]],
        sorted_files: list[str],
        max_total_chars: int,
        sample_count: int = 5,
    ) -> dict:
        """Palier 51+ docs : liste de fichiers + extraits pour un échantillon."""
        # Échantillonner à intervalles réguliers
        n = len(sorted_files)
        step = max(1, n // sample_count)
        sampled_files = [sorted_files[i] for i in range(0, n, step)][:sample_count]

        budget_per_sample = max_total_chars // len(sampled_files)
        entries = []
        for source_file in sampled_files:
            excerpt = ""
            for chunk in chunks_by_file[source_file]:
                remaining = budget_per_sample - len(excerpt)
                if remaining <= 0:
                    break
                excerpt += chunk.text[:remaining]
            if excerpt:
                entries.append({"source_file": source_file, "text": excerpt.strip()})

        return {
            "tier": "sampled",
            "num_documents": n,
            "entries": entries,
            "all_filenames": sorted_files,
        }

    def get_full_text(self) -> str:
        """Retourne le texte complet du corpus."""
        return "\n\n---\n\n".join(c.text for c in self.chunks)

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "source_files": self.source_files,
            "chunks": [
                {
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "source_file": c.source_file,
                    "chunk_index": c.chunk_index,
                    "token_estimate": c.token_estimate,
                }
                for c in self.chunks
            ],
        }


class CorpusExtractor:
    """Extraction et structuration du corpus documentaire."""

    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 300):
        """
        Args:
            chunk_size: Taille cible des blocs en caractères (~750 tokens).
            chunk_overlap: Chevauchement entre blocs en caractères (~75 tokens).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_corpus(self, corpus_dir: Path) -> StructuredCorpus:
        """Extrait et structure l'ensemble du corpus depuis un dossier."""
        corpus = StructuredCorpus()
        files = sorted(corpus_dir.iterdir())

        for file_path in files:
            if file_path.is_dir() or file_path.name.startswith(".") or file_path.suffix == ".json":
                continue

            result = extract(file_path)
            corpus.extractions.append(result)

            if result.status == "failed" or not result.text.strip():
                logger.warning(f"Extraction échouée ou vide pour {file_path.name}")
                continue

            corpus.source_files.append(file_path.name)
            chunks = self._split_text(result.text, file_path.name)
            corpus.chunks.extend(chunks)

        corpus.total_chunks = len(corpus.chunks)
        corpus.total_tokens = sum(c.token_estimate for c in corpus.chunks)

        logger.info(
            f"Corpus structuré : {len(corpus.source_files)} fichiers, "
            f"{corpus.total_chunks} blocs, ~{corpus.total_tokens} tokens"
        )
        return corpus

    def _split_text(self, text: str, source_file: str) -> list[CorpusChunk]:
        """Découpe un texte en blocs avec chevauchement."""
        chunks = []
        if len(text) <= self.chunk_size:
            chunks.append(CorpusChunk(
                text=text,
                source_file=source_file,
                chunk_index=0,
                char_count=len(text),
                token_estimate=len(text) // 4,
            ))
            return chunks

        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + self.chunk_size

            # Couper au dernier saut de ligne ou espace avant la fin
            if end < len(text):
                last_newline = text.rfind("\n", start, end)
                if last_newline > start + self.chunk_size // 2:
                    end = last_newline
                else:
                    last_space = text.rfind(" ", start, end)
                    if last_space > start + self.chunk_size // 2:
                        end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(CorpusChunk(
                    text=chunk_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    char_count=len(chunk_text),
                    token_estimate=len(chunk_text) // 4,
                ))
                chunk_index += 1

            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break

        return chunks
