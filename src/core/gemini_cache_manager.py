"""Gestionnaire du cycle de vie des caches Gemini pour Orchestr'IA — Phase 5.

Implémente le context caching explicite via l'API google.genai.caches
pour réduire les coûts de génération sur les grands corpus (~85% d'économie).

Contraintes API (non contournables) :
  C1 : system_instruction doit être inclus dans le cache à la création.
  C2 : tools incompatibles avec cached_content (erreur 400 sinon).
  C3 : contenu du cache immutable après création.
  C4 : minimum 2 048 tokens pour activer le cache.
  C5 : toujours utiliser le caching EXPLICITE (caching implicite instable).
  C6 : repricing long-context si input > 200 000 tokens.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from src.utils.token_counter import count_tokens

try:
    from google.genai import types as genai_types
except ImportError:
    genai_types = None  # type: ignore

logger = logging.getLogger("orchestria")

# Seuil minimum de tokens pour activer le cache (contrainte API)
_MIN_CACHE_TOKENS = 2048

# Marge de renouvellement TTL (en secondes)
_TTL_RENEWAL_MARGIN = 1800  # 30 minutes


class CacheTooSmallError(Exception):
    """Levée si le corpus est trop petit pour le context caching (< 2048 tokens)."""
    pass


class GeminiCacheManager:
    """Gestion du cycle de vie des caches Gemini Context Caching.

    Ne doit jamais être appelé directement depuis les pages Streamlit —
    passer exclusivement par l'orchestrateur.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client = None

    def _get_client(self):
        """Initialise le client Google genai (lazy)."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # ──────────────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────────────

    def create_corpus_cache(
        self,
        project_id: str,
        corpus_xml: str,
        system_prompt: str,
        model: str,
        ttl: int = 7200,
    ) -> str:
        """Crée un cache contenant le corpus XML et le system_prompt.

        Args:
            project_id: Identifiant du projet (utilisé dans les logs).
            corpus_xml: Corpus formaté en XML (voir format_corpus_xml).
            system_prompt: Instruction système stable pour ce projet.
            model: Modèle Gemini cible (ex: "gemini-3.1-pro-preview").
            ttl: Durée de vie du cache en secondes (défaut : 7200s = 2h).

        Returns:
            Nom du cache créé (ex: "cachedContents/abc123").

        Raises:
            CacheTooSmallError: Si le corpus est trop petit (< 2048 tokens).
        """
        token_count = count_tokens(corpus_xml + system_prompt)
        if token_count < _MIN_CACHE_TOKENS:
            raise CacheTooSmallError(
                f"Corpus trop petit pour le context caching : {token_count} tokens "
                f"(minimum requis : {_MIN_CACHE_TOKENS})."
            )

        client = self._get_client()

        cache = client.caches.create(
            model=model,
            config=genai_types.CreateCachedContentConfig(
                system_instruction=system_prompt,
                contents=[corpus_xml],
                ttl=f"{ttl}s",
            ),
        )

        cache_name = cache.name
        logger.info(
            f"[{project_id}] Cache Gemini créé : {cache_name} "
            f"({token_count} tokens, TTL={ttl}s)"
        )
        return cache_name

    def get_or_create_cache(
        self,
        project_id: str,
        corpus_xml: str,
        system_prompt: str,
        model: str,
        ttl: int = 7200,
        existing_cache_name: Optional[str] = None,
    ) -> str:
        """Vérifie ou crée un cache pour ce projet.

        Si un cache valide existe avec suffisamment de TTL restant, le retourne.
        Si TTL restant <= 30 min, prolonge le TTL.
        Sinon, crée un nouveau cache.

        Args:
            project_id: Identifiant du projet.
            corpus_xml: Corpus XML à mettre en cache.
            system_prompt: Instruction système stable.
            model: Modèle Gemini cible.
            ttl: TTL en secondes pour un nouveau cache.
            existing_cache_name: Nom du cache existant (depuis ProjectState.cache_id).

        Returns:
            Nom du cache (existant ou nouveau).
        """
        if existing_cache_name:
            try:
                client = self._get_client()
                cache = client.caches.get(name=existing_cache_name)

                # Calculer le TTL restant
                remaining = self._remaining_ttl_seconds(cache)

                if remaining > _TTL_RENEWAL_MARGIN:
                    logger.info(
                        f"[{project_id}] Réutilisation du cache existant : "
                        f"{existing_cache_name} (TTL restant : {remaining:.0f}s)"
                    )
                    return existing_cache_name
                elif remaining > 0:
                    logger.info(
                        f"[{project_id}] TTL cache presque expiré ({remaining:.0f}s). "
                        f"Prolongation..."
                    )
                    self.extend_cache_ttl(existing_cache_name, ttl)
                    return existing_cache_name
                else:
                    logger.info(
                        f"[{project_id}] Cache expiré : {existing_cache_name}. "
                        "Recréation..."
                    )
            except Exception as e:
                logger.warning(
                    f"[{project_id}] Cache introuvable ou erreur : {e}. "
                    "Création d'un nouveau cache."
                )

        return self.create_corpus_cache(project_id, corpus_xml, system_prompt, model, ttl)

    def extend_cache_ttl(self, cache_name: str, ttl: int) -> None:
        """Prolonge le TTL d'un cache existant.

        Args:
            cache_name: Nom du cache (ex: "cachedContents/abc123").
            ttl: Nouveau TTL en secondes.
        """
        client = self._get_client()
        client.caches.update(
            name=cache_name,
            config=genai_types.UpdateCachedContentConfig(ttl=f"{ttl}s"),
        )
        logger.info(f"TTL du cache {cache_name} prolongé à {ttl}s.")

    def delete_cache(self, cache_name: str) -> None:
        """Supprime un cache explicitement.

        À appeler en fin de projet ou si le corpus est modifié après la
        création du cache.

        Args:
            cache_name: Nom du cache à supprimer.
        """
        try:
            client = self._get_client()
            client.caches.delete(name=cache_name)
            logger.info(f"Cache Gemini supprimé : {cache_name}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le cache {cache_name} : {e}")

    def estimate_cache_cost(
        self,
        corpus_tokens: int,
        num_sections: int,
        ttl_hours: float,
        model: str,
    ) -> dict:
        """Estime les coûts avec et sans context caching.

        Args:
            corpus_tokens: Nombre de tokens dans le corpus.
            num_sections: Nombre de sections à générer.
            ttl_hours: Durée de vie du cache en heures.
            model: Modèle Gemini utilisé.

        Returns:
            Dictionnaire avec cost_without_cache, cost_with_cache, savings_usd,
            savings_percent, storage_cost, break_even_sections.
        """
        from src.utils.config import load_model_pricing
        pricing = load_model_pricing()
        model_pricing = pricing.get("google", {}).get(model, {})

        input_rate = model_pricing.get("input", 2.00)
        input_cached_rate = model_pricing.get("input_cached", 0.20)
        storage_rate = model_pricing.get("cache_storage_per_hour", 0.50)

        # Coût sans cache : chaque section relit l'intégralité du corpus
        cost_without_cache = (corpus_tokens / 1_000_000) * input_rate * num_sections

        # Coût avec cache :
        # - Création : 1 lecture standard du corpus
        # - Lectures : num_sections × corpus_tokens au tarif caché
        # - Stockage : tokens × heures × tarif_stockage
        cost_creation = (corpus_tokens / 1_000_000) * input_rate
        cost_reads = (corpus_tokens / 1_000_000) * input_cached_rate * num_sections
        storage_cost = (corpus_tokens / 1_000_000) * storage_rate * ttl_hours
        cost_with_cache = cost_creation + cost_reads + storage_cost

        savings_usd = cost_without_cache - cost_with_cache
        savings_percent = (savings_usd / cost_without_cache * 100) if cost_without_cache > 0 else 0.0

        # Break-even : à partir de combien de sections le cache est rentable ?
        cost_per_section_no_cache = (corpus_tokens / 1_000_000) * input_rate
        cost_per_section_with_cache = (corpus_tokens / 1_000_000) * input_cached_rate
        savings_per_section = cost_per_section_no_cache - cost_per_section_with_cache

        if savings_per_section > 0:
            break_even = int((cost_creation + storage_cost) / savings_per_section) + 1
        else:
            break_even = 999  # Cache jamais rentable

        return {
            "cost_without_cache": round(cost_without_cache, 4),
            "cost_with_cache": round(cost_with_cache, 4),
            "savings_usd": round(savings_usd, 4),
            "savings_percent": round(savings_percent, 1),
            "storage_cost": round(storage_cost, 4),
            "break_even_sections": break_even,
        }

    def should_use_cache(
        self,
        corpus_tokens: int,
        num_sections: int,
        model: str,
    ) -> bool:
        """Retourne True si l'utilisation du cache est rentable pour ce projet.

        Args:
            corpus_tokens: Nombre de tokens dans le corpus.
            num_sections: Nombre de sections à générer.
            model: Modèle Gemini utilisé.

        Returns:
            True si corpus >= 2048 tokens ET num_sections >= break_even.
        """
        if corpus_tokens < _MIN_CACHE_TOKENS:
            return False

        estimate = self.estimate_cache_cost(
            corpus_tokens=corpus_tokens,
            num_sections=num_sections,
            ttl_hours=2.0,
            model=model,
        )
        return num_sections >= estimate["break_even_sections"]

    # ──────────────────────────────────────────────────────────────────────────
    # Méthodes utilitaires
    # ──────────────────────────────────────────────────────────────────────────

    def _remaining_ttl_seconds(self, cache) -> float:
        """Calcule le TTL restant d'un cache en secondes."""
        try:
            expire_time = cache.expire_time
            if expire_time is None:
                return 0.0
            now = datetime.now(timezone.utc)
            if hasattr(expire_time, "timestamp"):
                remaining = expire_time.timestamp() - now.timestamp()
            else:
                # Cas où expire_time est un proto Timestamp
                remaining = float(expire_time.seconds) - now.timestamp()
            return max(0.0, remaining)
        except Exception:
            return 0.0


def format_corpus_xml(corpus_data: list[dict]) -> str:
    """Formate le corpus en XML structuré pour l'injection dans le cache.

    Format attendu pour corpus_data :
        [
            {
                "id": "001",
                "title": "Titre du document",
                "year": "2024",
                "type": "rapport",
                "chunks": [
                    {
                        "id": "001_001",
                        "page": "3",
                        "section": "Introduction",
                        "content": "Texte du chunk..."
                    },
                    ...
                ]
            },
            ...
        ]

    Args:
        corpus_data: Liste de documents avec leurs chunks.

    Returns:
        Chaîne XML structurée prête pour le cache Gemini.
    """
    import xml.etree.ElementTree as ET

    root = ET.Element("corpus")

    for doc in corpus_data:
        doc_elem = ET.SubElement(root, "document")
        doc_elem.set("id", str(doc.get("id", "")))
        doc_elem.set("title", str(doc.get("title", "")))
        doc_elem.set("year", str(doc.get("year", "")))
        doc_elem.set("type", str(doc.get("type", "")))

        for chunk in doc.get("chunks", []):
            chunk_elem = ET.SubElement(doc_elem, "chunk")
            chunk_elem.set("id", str(chunk.get("id", "")))
            chunk_elem.set("page", str(chunk.get("page", "")))
            chunk_elem.set("section", str(chunk.get("section", "")))
            chunk_elem.text = str(chunk.get("content", ""))

    # Sérialisation avec déclaration XML
    return ET.tostring(root, encoding="unicode", xml_declaration=False)
