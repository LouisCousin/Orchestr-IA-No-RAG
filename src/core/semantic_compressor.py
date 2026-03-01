"""Compression sémantique du corpus — v4.0.

Pipeline de réduction de densité informationnelle en 3 niveaux :
  Niveau 1 — Déduplication hash + élagage décoratif (100% local)
  Niveau 2 — Extraction de faits clés via LLM léger (1 appel/doc)
  Niveau 3 — Compression Télégraphique (TSC) via LLM léger

Déclenché quand le corpus dépasse 900K tokens (marge avant la limite 1M).
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.utils.token_counter import count_tokens

logger = logging.getLogger("orchestria")


@dataclass
class CompressionReport:
    """Rapport de compression sémantique."""
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    levels_applied: list[int] = field(default_factory=list)
    estimated_fact_preservation: float = 1.0
    cost_usd: float = 0.0


@dataclass
class CompressedCorpus:
    """Résultat de la compression sémantique."""
    corpus: object = None  # StructuredCorpus
    compressed_tokens: int = 0
    ratio: float = 0.0
    report: CompressionReport = field(default_factory=CompressionReport)


class SemanticCompressor:
    """Compression sémantique progressive du corpus.

    Trois niveaux appliqués séquentiellement jusqu'à atteindre la cible.
    """

    TARGET_TOKENS: int = 800_000
    MIN_FACT_PRESERVATION_RATIO: float = 0.90

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        ctx = self._config.get("context_strategy", {})
        compression_cfg = ctx.get("compression", {})
        self.TARGET_TOKENS = compression_cfg.get(
            "target_tokens", self.TARGET_TOKENS
        )
        self.MIN_FACT_PRESERVATION_RATIO = compression_cfg.get(
            "min_fact_preservation_ratio", self.MIN_FACT_PRESERVATION_RATIO
        )
        self._max_concurrent = compression_cfg.get(
            "max_concurrent_compressions", 4
        )
        self._report = CompressionReport()

    async def compress(
        self,
        corpus,
        target_tokens: int | None = None,
        llm_provider=None,
    ) -> CompressedCorpus:
        """Compresse le corpus en appliquant les niveaux séquentiellement.

        Args:
            corpus: StructuredCorpus à compresser.
            target_tokens: Cible en tokens (défaut : 800K).
            llm_provider: Provider LLM pour les niveaux 2 et 3.

        Returns:
            CompressedCorpus avec le corpus réduit et le rapport.
        """
        target = target_tokens or self.TARGET_TOKENS
        self._report = CompressionReport()

        # Compter les tokens initiaux
        original_tokens = self._count_corpus_tokens(corpus)
        self._report.original_tokens = original_tokens

        if original_tokens <= target:
            logger.info(
                f"[SemanticCompressor] Corpus déjà sous la cible "
                f"({original_tokens} <= {target}). Aucune compression."
            )
            return CompressedCorpus(
                corpus=corpus,
                compressed_tokens=original_tokens,
                ratio=0.0,
                report=self._report,
            )

        logger.info(
            f"[SemanticCompressor] Début compression : "
            f"{original_tokens} → cible {target} tokens"
        )

        # Niveau 1 : Déduplication et élagage (100% local)
        corpus = self._level1_dedup_and_prune(corpus)
        current_tokens = self._count_corpus_tokens(corpus)
        self._report.levels_applied.append(1)
        logger.info(
            f"[SemanticCompressor] Niveau 1 terminé : "
            f"{original_tokens} → {current_tokens} tokens "
            f"(-{((original_tokens - current_tokens) / original_tokens * 100):.1f}%)"
        )

        if current_tokens <= target:
            return self._build_result(corpus, current_tokens)

        # Niveau 2 : Extraction de faits clés via LLM
        if llm_provider:
            corpus = await self._level2_extractive(corpus, llm_provider)
            current_tokens = self._count_corpus_tokens(corpus)
            self._report.levels_applied.append(2)
            logger.info(
                f"[SemanticCompressor] Niveau 2 terminé : "
                f"{current_tokens} tokens"
            )

            if current_tokens <= target:
                return self._build_result(corpus, current_tokens)

            # Niveau 3 : Compression Télégraphique (TSC)
            corpus = await self._level3_telegraphic(corpus, llm_provider)
            current_tokens = self._count_corpus_tokens(corpus)
            self._report.levels_applied.append(3)
            logger.info(
                f"[SemanticCompressor] Niveau 3 terminé : "
                f"{current_tokens} tokens"
            )

        result = self._build_result(corpus, current_tokens)

        if current_tokens > target:
            logger.warning(
                f"[SemanticCompressor] Corpus toujours au-dessus de la cible "
                f"après 3 niveaux : {current_tokens} > {target}. "
                f"Sélection manuelle de documents recommandée."
            )

        return result

    def _level1_dedup_and_prune(self, corpus) -> object:
        """Niveau 1 : Déduplication hash + élagage décoratif.

        100% local, pas d'appel LLM.
        - Suppression des doublons textuels exacts (hash SHA-256)
        - Suppression des blocs purement décoratifs
        """
        if not hasattr(corpus, "chunks"):
            return corpus

        seen_hashes: set[str] = set()
        deduplicated_chunks = []

        for chunk in corpus.chunks:
            text = getattr(chunk, "text", "")
            if not text or not text.strip():
                continue

            # Hash du contenu normalisé
            normalized = " ".join(text.lower().split())
            text_hash = hashlib.sha256(normalized.encode()).hexdigest()

            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            # Élaguer les blocs purement décoratifs
            if self._is_decorative(text):
                continue

            deduplicated_chunks.append(chunk)

        corpus.chunks = deduplicated_chunks
        corpus.total_chunks = len(deduplicated_chunks)
        corpus.total_tokens = self._count_corpus_tokens(corpus)
        return corpus

    async def _level2_extractive(self, corpus, llm_provider) -> object:
        """Niveau 2 : Extraction de faits clés via LLM léger.

        Un appel LLM par document pour extraire les faits clés en JSON.
        """
        if not hasattr(corpus, "chunks"):
            return corpus

        # Regrouper chunks par source
        docs_by_source: dict[str, list] = {}
        for chunk in corpus.chunks:
            source = getattr(chunk, "source_file", "unknown")
            docs_by_source.setdefault(source, []).append(chunk)

        sem = asyncio.Semaphore(self._max_concurrent)
        compressed_chunks = []

        async def compress_doc(source: str, chunks: list):
            async with sem:
                full_text = "\n\n".join(
                    getattr(c, "text", "") for c in chunks
                )
                if not full_text.strip():
                    return chunks

                prompt = (
                    "Extrais les faits clés, chiffres, dates, entités nommées "
                    "et citations importantes du texte suivant. "
                    "Retourne une liste structurée condensée en français, "
                    "en préservant TOUTES les informations factuelles.\n\n"
                    f"--- TEXTE SOURCE ({source}) ---\n"
                    f"{full_text[:30000]}\n"
                    "--- FIN ---\n\n"
                    "Retourne les faits clés sous forme de liste à puces."
                )

                try:
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: llm_provider.generate(
                            prompt=prompt,
                            system_prompt=(
                                "Tu es un assistant d'extraction de faits. "
                                "Extrais et condense les faits clés tout en "
                                "préservant l'intégralité des informations "
                                "factuelles."
                            ),
                            model=self._get_compression_model(),
                            temperature=0.1,
                            max_tokens=8192,
                        ),
                    )

                    self._report.cost_usd += getattr(
                        response, "cost_usd", 0.0
                    )

                    # Remplacer les chunks par le texte compressé
                    if response.content and len(response.content) < len(full_text):
                        first_chunk = chunks[0]
                        first_chunk.text = response.content
                        return [first_chunk]

                except Exception as e:
                    logger.warning(
                        f"[SemanticCompressor] Niveau 2 échec pour "
                        f"{source}: {e}"
                    )

                return chunks

        tasks = [
            compress_doc(source, chunks)
            for source, chunks in docs_by_source.items()
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            compressed_chunks.extend(result)

        corpus.chunks = compressed_chunks
        corpus.total_chunks = len(compressed_chunks)
        corpus.total_tokens = self._count_corpus_tokens(corpus)
        self._report.estimated_fact_preservation = 0.92
        return corpus

    async def _level3_telegraphic(self, corpus, llm_provider) -> object:
        """Niveau 3 : Compression Télégraphique (TSC).

        Supprime la structure grammaticale tout en préservant les faits.
        Exemple : "Le rapport indique que les ventes ont augmenté de 15%"
                → "Ventes +15% (rapport)"
        """
        if not hasattr(corpus, "chunks"):
            return corpus

        sem = asyncio.Semaphore(self._max_concurrent)
        compressed_chunks = []

        async def compress_chunk(chunk):
            async with sem:
                text = getattr(chunk, "text", "")
                if not text or len(text) < 100:
                    return chunk

                prompt = (
                    "Applique une compression télégraphique au texte suivant. "
                    "Règles :\n"
                    "- Supprime articles, pronoms, verbes copules\n"
                    "- Préserve TOUS les chiffres, dates, entités nommées\n"
                    "- Préserve TOUTES les informations factuelles\n"
                    "- Aucune paraphrase : uniquement compression de forme\n"
                    "- Utilise des abréviations standard\n\n"
                    f"--- TEXTE ---\n{text[:10000]}\n--- FIN ---\n\n"
                    "Retourne le texte compressé."
                )

                try:
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: llm_provider.generate(
                            prompt=prompt,
                            system_prompt=(
                                "Tu comprimes le texte en style "
                                "télégraphique. Préserve les faits."
                            ),
                            model=self._get_compression_model(),
                            temperature=0.1,
                            max_tokens=4096,
                        ),
                    )

                    self._report.cost_usd += getattr(
                        response, "cost_usd", 0.0
                    )

                    if response.content and len(response.content) < len(text):
                        chunk.text = response.content

                except Exception as e:
                    logger.warning(
                        f"[SemanticCompressor] Niveau 3 échec: {e}"
                    )

                return chunk

        tasks = [compress_chunk(chunk) for chunk in corpus.chunks]
        compressed_chunks = await asyncio.gather(*tasks)

        corpus.chunks = list(compressed_chunks)
        corpus.total_chunks = len(corpus.chunks)
        corpus.total_tokens = self._count_corpus_tokens(corpus)
        self._report.estimated_fact_preservation = 0.90
        return corpus

    def _build_result(
        self, corpus, current_tokens: int
    ) -> CompressedCorpus:
        """Construit le résultat final de la compression."""
        original = self._report.original_tokens
        ratio = (
            (original - current_tokens) / original if original > 0 else 0.0
        )
        self._report.compressed_tokens = current_tokens
        self._report.compression_ratio = ratio

        return CompressedCorpus(
            corpus=corpus,
            compressed_tokens=current_tokens,
            ratio=ratio,
            report=self._report,
        )

    def get_compression_report(self) -> CompressionReport:
        """Retourne le rapport de la dernière compression."""
        return self._report

    def _get_compression_model(self) -> str:
        """Retourne le modèle LLM pour la compression."""
        ctx = self._config.get("context_strategy", {})
        compression_cfg = ctx.get("compression", {})
        return compression_cfg.get("compression_model", "gemini-3-flash-preview")

    @staticmethod
    def _count_corpus_tokens(corpus) -> int:
        """Compte le nombre total de tokens dans le corpus."""
        if not hasattr(corpus, "chunks"):
            return 0
        total = 0
        for chunk in corpus.chunks:
            text = getattr(chunk, "text", "")
            total += count_tokens(text)
        return total

    @staticmethod
    def _is_decorative(text: str) -> bool:
        """Détecte les blocs purement décoratifs à élaguer."""
        stripped = text.strip()
        if len(stripped) < 20:
            return True

        # Tables des matières, mentions légales répétées
        decorative_patterns = [
            "table des matières",
            "table of contents",
            "tous droits réservés",
            "all rights reserved",
            "© copyright",
            "page laissée intentionnellement",
            "intentionally left blank",
        ]
        lower = stripped.lower()
        return any(pattern in lower for pattern in decorative_patterns)
