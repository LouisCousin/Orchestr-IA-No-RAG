"""Calcul d'embeddings locaux multilingues pour le pipeline RAG.

Phase 2.5 : utilise intfloat/multilingual-e5-large via FastEmbed (ONNX Quantized).
~300 Mo RAM vs ~1.5 Go avec sentence-transformers, vitesse x4.

IMPORTANT : Le modèle E5 exige des préfixes différents :
  - Documents : 'passage: '
  - Requêtes  : 'query: '
L'omission de ces préfixes dégrade la qualité de 15-20%.

Note : sentence-transformers reste installé pour le Reranker (cross-encoder).
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("orchestria")

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_CACHE_DIR = "./models"


class LocalEmbedder:
    """Calcul d'embeddings en local avec FastEmbed (ONNX Quantized)."""

    _instance: Optional["LocalEmbedder"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        self._model_name = model_name or DEFAULT_MODEL
        self._cache_dir = cache_dir or os.environ.get("ORCHESTRIA_MODELS_DIR", DEFAULT_CACHE_DIR)
        self._model = None

    def _load_model(self):
        """Charge le modèle FastEmbed de manière paresseuse."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError(
                    "fastembed est requis pour les embeddings locaux. "
                    "Installez-le avec : pip install fastembed"
                )
            logger.info(f"Chargement du modèle d'embeddings FastEmbed : {self._model_name}")
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                threads=2,
            )
            logger.info("Modèle d'embedding FastEmbed chargé (ONNX Quantized)")
        return self._model

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> "LocalEmbedder":
        """Retourne l'instance singleton (thread-safe, le modèle est lourd à charger)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name, cache_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Réinitialise le singleton (utile pour les tests)."""
        cls._instance = None

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Encode une liste de textes de corpus (avec préfixe 'passage:').

        Args:
            texts: Liste de textes à encoder.
            batch_size: Taille des lots pour l'inférence (défaut: 16, conservateur pour RAM).

        Returns:
            Liste de vecteurs normalisés (float Python natif, pas np.float32).
        """
        model = self._load_model()
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = model.embed(prefixed, batch_size=batch_size)
        return [list(float(x) for x in e) for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Encode une requête de recherche (avec préfixe 'query:').

        Args:
            query: Texte de la requête.

        Returns:
            Vecteur normalisé (float Python natif).
        """
        model = self._load_model()
        embeddings = list(model.embed([f"query: {query}"]))
        return [float(x) for x in embeddings[0]]

    @property
    def dimension(self) -> int:
        """Dimension des vecteurs produits."""
        return 1024
