"""Moteur RAG simplifié avec ChromaDB pour Orchestr'IA (Phase 2)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestria")


@dataclass
class RAGResult:
    """Résultat d'une recherche RAG pour une section."""
    section_id: str
    section_title: str
    chunks: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    avg_score: float = 0.0
    num_relevant: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "num_chunks": len(self.chunks),
            "avg_score": round(self.avg_score, 4),
            "num_relevant": self.num_relevant,
            "total_tokens": self.total_tokens,
        }


class RAGEngine:
    """Moteur RAG simplifié utilisant ChromaDB comme base vectorielle locale.

    Découpe les documents en blocs, les indexe via embeddings, et effectue
    des recherches par similarité pour chaque section du plan.
    """

    def __init__(
        self,
        collection_name: str = "orchestria_corpus",
        persist_dir: Optional[Path] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 7,
        relevance_threshold: float = 0.3,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self._client = None
        self._collection = None

    def _get_client(self):
        """Initialise le client ChromaDB (lazy)."""
        if self._client is None:
            import chromadb
            if self.persist_dir:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            else:
                self._client = chromadb.Client()
        return self._client

    def _get_collection(self):
        """Récupère ou crée la collection ChromaDB."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_corpus(self, extractions: list[dict]) -> int:
        """Indexe le corpus dans ChromaDB.

        Args:
            extractions: Liste de dicts avec les clés 'text', 'source_file',
                et optionnellement 'page_count', 'hash_text'.

        Returns:
            Nombre de blocs indexés.
        """
        collection = self._get_collection()

        # Vider la collection existante si elle contient des données
        existing = collection.count()
        if existing > 0:
            # Récupérer tous les IDs et les supprimer
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            logger.info(f"Collection existante vidée ({existing} blocs supprimés)")

        documents = []
        metadatas = []
        ids = []
        chunk_index = 0

        for extraction in extractions:
            text = extraction.get("text", "")
            source = extraction.get("source_file", "unknown")

            if not text.strip():
                continue

            chunks = self._split_text(text)
            for i, chunk_text in enumerate(chunks):
                doc_id = f"{source}_chunk_{i:04d}"
                documents.append(chunk_text)
                metadatas.append({
                    "source_file": source,
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                    "token_estimate": len(chunk_text) // 4,
                })
                ids.append(doc_id)
                chunk_index += 1

        if documents:
            # ChromaDB indexe par lots de 5000 max
            batch_size = 5000
            for start in range(0, len(documents), batch_size):
                end = min(start + batch_size, len(documents))
                collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )

        logger.info(f"Corpus indexé dans ChromaDB : {chunk_index} blocs depuis {len(extractions)} documents")
        return chunk_index

    def search(self, query: str, top_k: Optional[int] = None) -> RAGResult:
        """Recherche les blocs les plus pertinents pour une requête.

        Args:
            query: Texte de la requête (titre de section + description).
            top_k: Nombre de résultats à retourner (défaut: self.top_k).

        Returns:
            RAGResult avec les blocs trouvés et les scores.
        """
        top_k = top_k or self.top_k
        collection = self._get_collection()

        if collection.count() == 0:
            return RAGResult(section_id="", section_title=query)

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        scores = []
        total_tokens = 0

        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB retourne des distances cosinus (0 = identique, 2 = opposé)
                # Convertir en score de similarité (1 = identique, 0 = opposé)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1.0 - (distance / 2.0)

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                token_estimate = metadata.get("token_estimate", len(doc) // 4)
                total_tokens += token_estimate

                chunks.append({
                    "text": doc,
                    "source_file": metadata.get("source_file", "unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity": round(similarity, 4),
                    "token_estimate": token_estimate,
                })
                scores.append(similarity)

        num_relevant = sum(1 for s in scores if s >= self.relevance_threshold)
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RAGResult(
            section_id="",
            section_title=query,
            chunks=chunks,
            scores=scores,
            avg_score=avg_score,
            num_relevant=num_relevant,
            total_tokens=total_tokens,
        )

    def search_for_section(self, section_id: str, section_title: str, section_description: str = "") -> RAGResult:
        """Recherche les blocs pertinents pour une section du plan.

        Args:
            section_id: Identifiant de la section.
            section_title: Titre de la section.
            section_description: Description optionnelle de la section.

        Returns:
            RAGResult avec les résultats de la recherche.
        """
        query = section_title
        if section_description:
            query = f"{section_title}. {section_description}"

        result = self.search(query)
        result.section_id = section_id
        result.section_title = section_title
        return result

    def get_corpus_chunks_for_section(
        self,
        section_id: str,
        section_title: str,
        section_description: str = "",
    ) -> list[dict]:
        """Retourne les chunks de corpus formatés pour injection dans le prompt.

        Compatibilité avec l'interface existante de CorpusExtractor.
        """
        result = self.search_for_section(section_id, section_title, section_description)
        return result.chunks

    def _split_text(self, text: str) -> list[str]:
        """Découpe un texte en blocs avec chevauchement.

        Les blocs font environ chunk_size tokens (estimés à 4 chars/token).
        """
        char_size = self.chunk_size * 4  # ~4 chars par token
        char_overlap = self.chunk_overlap * 4
        chunks = []

        if len(text) <= char_size:
            return [text.strip()] if text.strip() else []

        start = 0
        while start < len(text):
            end = start + char_size

            # Couper à une frontière naturelle
            if end < len(text):
                last_newline = text.rfind("\n", start, end)
                if last_newline > start + char_size // 2:
                    end = last_newline
                else:
                    last_space = text.rfind(" ", start, end)
                    if last_space > start + char_size // 2:
                        end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - char_overlap
            if start >= len(text) - char_overlap:
                break

        return chunks

    def reset(self) -> None:
        """Réinitialise la collection ChromaDB."""
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
        logger.info("Collection RAG réinitialisée")

    @property
    def indexed_count(self) -> int:
        """Retourne le nombre de blocs indexés."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0
