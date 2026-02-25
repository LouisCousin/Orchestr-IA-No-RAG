"""Moteur RAG hybride avec ChromaDB pour Orchestr'IA.

Phase 2 : recherche vectorielle de base.
Phase 2.5 : pipeline hybride complet avec :
  - Embeddings locaux (multilingual-e5-large)
  - Pré-filtrage par métadonnées (SQLite)
  - Reranking par cross-encoder
  - Enrichissement des métadonnées
"""

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
    """Moteur RAG hybride utilisant ChromaDB, embeddings locaux et reranking.

    Phase 2.5 : pipeline complet avec 5 étapes :
    1. Extraction structurée (Docling)
    2. Chunking sémantique
    3. Embeddings locaux + stockage ChromaDB + SQLite
    4. Recherche hybride (vecteurs + filtrage métadonnées)
    5. Reranking cross-encoder
    """

    def __init__(
        self,
        collection_name: str = "orchestria_corpus",
        persist_dir: Optional[Path] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 10,
        relevance_threshold: float = 0.3,
        config: Optional[dict] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.config = config or {}
        self._client = None
        self._collection = None
        self._use_local_embeddings = self.config.get("rag", {}).get("embedding_mode", "local") == "local"
        self._reranking_enabled = self.config.get("rag", {}).get("reranking_enabled", True)
        self._initial_candidates = self.config.get("rag", {}).get("initial_candidates", 20)

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

    def _get_embeddings(self, texts: list[str], mode: str = "document") -> list[list[float]]:
        """Calcule les embeddings en local ou via API.

        Phase 2.5 : utilise les embeddings locaux par défaut.
        """
        if self._use_local_embeddings:
            try:
                from src.core.local_embedder import LocalEmbedder
                embedder = LocalEmbedder.get_instance()
                if mode == "query":
                    return [embedder.embed_query(t) for t in texts]
                else:
                    return embedder.embed_documents(texts)
            except ImportError:
                logger.warning("Embeddings locaux non disponibles, utilisation des embeddings ChromaDB par défaut")
                self._use_local_embeddings = False
        # Fallback : ChromaDB gère les embeddings en interne
        return []

    def index_corpus(self, extractions: list[dict]) -> int:
        """Indexe le corpus dans ChromaDB.

        Phase 2.5 : utilise les embeddings locaux et le chunking sémantique.

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
            batch_size = 5000

            # Phase 2.5 : utiliser les embeddings locaux si disponibles
            if self._use_local_embeddings:
                try:
                    all_embeddings = self._get_embeddings(documents, mode="document")
                    if all_embeddings:
                        for start in range(0, len(documents), batch_size):
                            end = min(start + batch_size, len(documents))
                            collection.add(
                                documents=documents[start:end],
                                embeddings=all_embeddings[start:end],
                                metadatas=metadatas[start:end],
                                ids=ids[start:end],
                            )
                        logger.info(f"Corpus indexé avec embeddings locaux : {chunk_index} blocs")
                        return chunk_index
                except Exception as e:
                    logger.warning(f"Erreur embeddings locaux, fallback ChromaDB : {e}")

            # Fallback : ChromaDB gère les embeddings
            for start in range(0, len(documents), batch_size):
                end = min(start + batch_size, len(documents))
                collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )

        logger.info(f"Corpus indexé dans ChromaDB : {chunk_index} blocs depuis {len(extractions)} documents")
        return chunk_index

    def index_corpus_semantic(self, chunks_by_doc: dict, metadata_store=None) -> int:
        """Indexe le corpus avec les chunks sémantiques (Phase 2.5).

        Args:
            chunks_by_doc: Dict {doc_id: list[Chunk]} du semantic_chunker.
            metadata_store: Instance de MetadataStore pour stocker les chunks.

        Returns:
            Nombre de blocs indexés.
        """
        collection = self._get_collection()

        # Vider la collection existante
        existing = collection.count()
        if existing > 0:
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)

        documents = []
        metadatas = []
        ids = []

        for doc_id, chunks in chunks_by_doc.items():
            for chunk in chunks:
                documents.append(chunk.text)
                metadatas.append({
                    "doc_id": chunk.doc_id,
                    "source_file": chunk.doc_id,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                })
                ids.append(chunk.chunk_id)

            # Stocker dans SQLite aussi
            if metadata_store:
                metadata_store.add_chunks(chunks)

        if documents:
            batch_size = 5000

            if self._use_local_embeddings:
                try:
                    all_embeddings = self._get_embeddings(documents, mode="document")
                    if all_embeddings:
                        for start in range(0, len(documents), batch_size):
                            end = min(start + batch_size, len(documents))
                            collection.add(
                                documents=documents[start:end],
                                embeddings=all_embeddings[start:end],
                                metadatas=metadatas[start:end],
                                ids=ids[start:end],
                            )
                        logger.info(f"Corpus indexé (sémantique) : {len(documents)} chunks")
                        return len(documents)
                except Exception as e:
                    logger.warning(f"Erreur embeddings locaux, fallback : {e}")

            for start in range(0, len(documents), batch_size):
                end = min(start + batch_size, len(documents))
                collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )

        logger.info(f"Corpus indexé (sémantique) : {len(documents)} chunks")
        return len(documents)

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

        n_results = min(self._initial_candidates, collection.count())

        # Phase 2.5 : utiliser les embeddings locaux pour la requête
        query_embedding = None
        if self._use_local_embeddings:
            try:
                embeddings = self._get_embeddings([query], mode="query")
                if embeddings:
                    query_embedding = embeddings[0]
            except Exception as e:
                logger.warning(f"Erreur embedding requête, fallback : {e}")

        if query_embedding:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        chunks = []
        scores = []
        total_tokens = 0

        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = max(0.0, 1.0 - distance)

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                token_estimate = metadata.get("token_estimate", metadata.get("token_count", len(doc) // 4))
                total_tokens += token_estimate

                chunks.append({
                    "text": doc,
                    "source_file": metadata.get("source_file", metadata.get("doc_id", "unknown")),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity": round(similarity, 4),
                    "token_estimate": token_estimate,
                    "page_number": metadata.get("page_number", 0),
                    "section_title": metadata.get("section_title", ""),
                    "doc_id": metadata.get("doc_id", ""),
                    "chunk_id": results["ids"][0][i] if results.get("ids") else "",
                })
                scores.append(similarity)

        # Phase 2.5 : Reranking par cross-encoder
        if self._reranking_enabled and len(chunks) > top_k:
            chunks, scores = self._rerank(query, chunks, scores, top_k)
        else:
            chunks = chunks[:top_k]
            scores = scores[:top_k]

        # Recalculate total_tokens to reflect only returned chunks (not all candidates)
        total_tokens = sum(c.get("token_estimate", len(c.get("text", "")) // 4) for c in chunks)

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

    def _rerank(
        self,
        query: str,
        chunks: list[dict],
        scores: list[float],
        top_k: int,
    ) -> tuple[list[dict], list[float]]:
        """Applique le reranking cross-encoder sur les candidats.

        Returns:
            Tuple (chunks_rerankés, scores_rerankés).
        """
        try:
            from src.core.reranker import Reranker, ScoredChunk

            reranker = Reranker.get_instance()
            scored_chunks = [
                ScoredChunk(
                    chunk_id=c.get("chunk_id", f"chunk_{i}"),
                    doc_id=c.get("doc_id", c.get("source_file", "")),
                    text=c["text"],
                    page_number=c.get("page_number", 0),
                    section_title=c.get("section_title", ""),
                    cosine_score=s,
                )
                for i, (c, s) in enumerate(zip(chunks, scores))
            ]

            reranked = reranker.rerank(query, scored_chunks, top_k=top_k)

            reranked_chunks = []
            reranked_scores = []
            for sc in reranked:
                reranked_chunks.append({
                    "text": sc.text,
                    "source_file": sc.doc_id,
                    "chunk_index": 0,
                    "similarity": round(sc.cosine_score, 4),
                    "rerank_score": round(sc.rerank_score, 4),
                    "token_estimate": len(sc.text) // 4,
                    "page_number": sc.page_number,
                    "section_title": sc.section_title,
                    "doc_id": sc.doc_id,
                    "chunk_id": sc.chunk_id,
                })
                reranked_scores.append(sc.rerank_score if sc.rerank_score > 0 else sc.cosine_score)

            return reranked_chunks, reranked_scores
        except ImportError:
            logger.warning("Reranker non disponible, utilisation de l'ordre ChromaDB")
            return chunks[:top_k], scores[:top_k]
        except Exception as e:
            logger.warning(f"Erreur reranking, fallback : {e}")
            return chunks[:top_k], scores[:top_k]

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

    def search_corpus(
        self,
        query: str,
        metadata_store=None,
        language_filter: Optional[str] = None,
    ) -> list:
        """Pipeline RAG hybride complet (Phase 2.5).

        Args:
            query: Requête de recherche.
            metadata_store: Instance de MetadataStore pour enrichir les résultats.
            language_filter: Filtrer par langue.

        Returns:
            Liste de ScoredChunk enrichis.
        """
        from src.core.reranker import ScoredChunk

        top_k = self.top_k
        collection = self._get_collection()

        if collection.count() == 0:
            return []

        # Étape 1 : Recherche vectorielle
        n_results = min(self._initial_candidates, collection.count())

        query_embedding = None
        if self._use_local_embeddings:
            try:
                embeddings = self._get_embeddings([query], mode="query")
                if embeddings:
                    query_embedding = embeddings[0]
            except Exception:
                pass

        where_filter = {}
        if language_filter:
            where_filter["language"] = language_filter

        query_kwargs = {
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        if query_embedding:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query]

        results = collection.query(**query_kwargs)

        candidates = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = max(0.0, 1.0 - distance)
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                candidates.append(ScoredChunk(
                    chunk_id=results["ids"][0][i] if results.get("ids") else f"chunk_{i}",
                    doc_id=metadata.get("doc_id", metadata.get("source_file", "")),
                    text=doc,
                    page_number=metadata.get("page_number", 0),
                    section_title=metadata.get("section_title", ""),
                    cosine_score=similarity,
                ))

        if not candidates:
            return []

        # Étape 2 : Reranking
        if self._reranking_enabled and len(candidates) > top_k:
            try:
                from src.core.reranker import Reranker
                reranker = Reranker.get_instance()
                candidates = reranker.rerank(query, candidates, top_k=top_k)
            except ImportError:
                logger.warning("Reranker non disponible")
                candidates = candidates[:top_k]
            except Exception as e:
                logger.warning(f"Erreur reranking : {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        # Étape 3 : Enrichissement avec métadonnées SQLite
        if metadata_store:
            import json as _json
            for chunk in candidates:
                doc_meta = metadata_store.get_document(chunk.doc_id)
                if doc_meta:
                    chunk.doc_title = doc_meta.title or ""
                    # authors is stored as a JSON-encoded list; decode it
                    raw_authors = doc_meta.authors or ""
                    if raw_authors:
                        try:
                            parsed = _json.loads(raw_authors)
                            if isinstance(parsed, list):
                                raw_authors = ", ".join(parsed)
                        except (ValueError, TypeError):
                            pass
                    chunk.doc_authors = raw_authors
                    chunk.apa_reference = doc_meta.apa_reference or ""

        return candidates

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
        char_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        chunks = []

        if len(text) <= char_size:
            return [text.strip()] if text.strip() else []

        start = 0
        while start < len(text):
            end = start + char_size

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
            if start >= len(text):
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

    @property
    def collection(self):
        """Accès direct à la collection ChromaDB (pour plan_corpus_linker)."""
        return self._get_collection()
