"""Cross-encoder pour le reranking post-retrieval.

Phase 2.5 : utilise cross-encoder/ms-marco-MiniLM-L-12-v2 pour reclasser
les candidats retournés par la recherche vectorielle ChromaDB.
Le reranking améliore la précision de 10-20% par rapport à la recherche
vectorielle seule.

Phase 4 (Perf) : détection automatique du device (cuda > mps > cpu).
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("orchestria")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@dataclass
class ScoredChunk:
    """Chunk avec scores de pertinence."""
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    section_title: str
    cosine_score: float          # Score de similarité ChromaDB
    rerank_score: float = 0.0    # Score du cross-encoder (0 si désactivé)
    # Métadonnées enrichies (remplies après recherche SQLite)
    doc_title: str = ""
    doc_authors: str = ""
    apa_reference: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "cosine_score": round(self.cosine_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "doc_title": self.doc_title,
            "doc_authors": self.doc_authors,
            "apa_reference": self.apa_reference,
        }


class Reranker:
    """Cross-encoder pour le reranking post-retrieval."""

    _instance: Optional["Reranker"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        self._model_name = model_name or DEFAULT_RERANKER_MODEL
        self._cache_dir = cache_dir or os.environ.get("ORCHESTRIA_MODELS_DIR", "./models")
        self._model = None

    def _load_model(self):
        """Charge le cross-encoder de manière paresseuse sur le meilleur device disponible."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                device = self._detect_device()
                logger.info(f"Chargement du modèle de reranking : {self._model_name} sur {device}")
                self._model = CrossEncoder(self._model_name, device=device)
                logger.info(f"Modèle de reranking chargé sur {device}.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers est requis pour le reranking. "
                    "Installez-le avec : pip install sentence-transformers"
                )
        return self._model

    @staticmethod
    def _detect_device() -> str:
        """Détecte le meilleur device disponible pour l'inférence."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> "Reranker":
        """Retourne l'instance singleton (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name, cache_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Réinitialise le singleton (utile pour les tests)."""
        cls._instance = None

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """Re-classe les candidats par pertinence et retourne le top-K.

        Args:
            query: Requête de recherche.
            candidates: Liste de ScoredChunk candidats.
            top_k: Nombre maximum de résultats à retourner.

        Returns:
            Liste triée par rerank_score décroissant, limitée à top_k.
        """
        if not candidates:
            return []

        model = self._load_model()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_k]


def build_context(chunks: list[ScoredChunk]) -> str:
    """Formate les ScoredChunk en blocs de contexte pour le prompt de génération.

    Args:
        chunks: Liste de ScoredChunk triés par pertinence.

    Returns:
        Texte formaté avec métadonnées, prêt pour injection dans le prompt.
    """
    if not chunks:
        return "Aucun bloc de corpus pertinent trouvé."

    blocks = []
    for i, chunk in enumerate(chunks):
        ref = chunk.apa_reference or chunk.doc_title or chunk.doc_id
        block = f"""--- SOURCE {i + 1} ---
Référence : {ref}
Page : {chunk.page_number}
Section : {chunk.section_title}
Contenu :
{chunk.text}
--- FIN SOURCE {i + 1} ---"""
        blocks.append(block)
    return "\n\n".join(blocks)
