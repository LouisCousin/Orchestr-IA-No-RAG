"""Tests unitaires pour le SemanticCompressor — v4.0."""

import pytest

from src.core.semantic_compressor import (
    CompressedCorpus,
    CompressionReport,
    SemanticCompressor,
)


class MockChunk:
    """Chunk de corpus fictif pour les tests."""

    def __init__(self, text, source_file="test.pdf", page_number=1, section_title=""):
        self.text = text
        self.source_file = source_file
        self.page_number = page_number
        self.section_title = section_title


class MockCorpus:
    """Corpus fictif pour les tests."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_chunks = len(chunks)
        self.total_tokens = 0


class TestSemanticCompressor:
    """Tests de la compression sémantique."""

    def setup_method(self):
        self.compressor = SemanticCompressor()

    @pytest.mark.asyncio
    async def test_compress_below_target(self):
        """Corpus déjà sous la cible → aucune compression."""
        corpus = MockCorpus([
            MockChunk("Un court texte."),
        ])

        result = await self.compressor.compress(corpus, target_tokens=1_000_000)
        assert isinstance(result, CompressedCorpus)
        assert result.ratio == 0.0
        assert result.report.levels_applied == []

    @pytest.mark.asyncio
    async def test_level1_deduplication(self):
        """Niveau 1 : les doublons exacts sont supprimés."""
        chunks = [
            MockChunk("Texte identique qui se répète", "doc1.pdf"),
            MockChunk("Texte identique qui se répète", "doc1.pdf"),  # doublon
            MockChunk("Texte différent avec du contenu unique", "doc2.pdf"),
        ]
        corpus = MockCorpus(chunks)

        result = self.compressor._level1_dedup_and_prune(corpus)
        assert len(result.chunks) == 2  # Le doublon supprimé

    @pytest.mark.asyncio
    async def test_level1_decorative_removal(self):
        """Niveau 1 : les blocs décoratifs sont supprimés."""
        chunks = [
            MockChunk("Contenu réel avec des informations importantes", "doc1.pdf"),
            MockChunk("Table des matières", "doc1.pdf"),  # décoratif
            MockChunk("Tous droits réservés © 2024", "doc1.pdf"),  # décoratif
            MockChunk("x", "doc1.pdf"),  # trop court
        ]
        corpus = MockCorpus(chunks)

        result = self.compressor._level1_dedup_and_prune(corpus)
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "Contenu réel avec des informations importantes"

    def test_is_decorative_short_text(self):
        """Texte trop court (< 20 chars) est décoratif."""
        assert SemanticCompressor._is_decorative("Court") is True
        assert SemanticCompressor._is_decorative("") is True

    def test_is_decorative_toc(self):
        """Table des matières détectée."""
        assert SemanticCompressor._is_decorative("Voici la table des matières du document") is True

    def test_is_decorative_normal_text(self):
        """Texte normal pas décoratif."""
        assert SemanticCompressor._is_decorative(
            "Le marché a connu une croissance de 15% en 2024."
        ) is False

    def test_count_corpus_tokens(self):
        """Le comptage de tokens fonctionne."""
        long_text = (
            "Un texte avec plusieurs mots pour tester le comptage de tokens. "
            "Ce texte doit être suffisamment long pour que le compteur heuristique "
            "retourne un résultat positif même sans tiktoken installé."
        )
        corpus = MockCorpus([
            MockChunk(long_text),
        ])
        tokens = SemanticCompressor._count_corpus_tokens(corpus)
        assert tokens > 0

    def test_count_corpus_tokens_empty(self):
        """Corpus vide → 0 tokens."""
        assert SemanticCompressor._count_corpus_tokens(None) == 0

    def test_compression_report_defaults(self):
        """Le rapport de compression a des valeurs par défaut correctes."""
        report = CompressionReport()
        assert report.original_tokens == 0
        assert report.compressed_tokens == 0
        assert report.compression_ratio == 0.0
        assert report.levels_applied == []
        assert report.estimated_fact_preservation == 1.0
        assert report.cost_usd == 0.0

    def test_get_compression_report(self):
        """get_compression_report retourne le dernier rapport."""
        report = self.compressor.get_compression_report()
        assert isinstance(report, CompressionReport)

    def test_custom_config(self):
        """Configuration personnalisée prise en compte."""
        config = {
            "context_strategy": {
                "compression": {
                    "target_tokens": 500_000,
                    "min_fact_preservation_ratio": 0.95,
                    "max_concurrent_compressions": 2,
                }
            }
        }
        compressor = SemanticCompressor(config=config)
        assert compressor.TARGET_TOKENS == 500_000
        assert compressor.MIN_FACT_PRESERVATION_RATIO == 0.95
        assert compressor._max_concurrent == 2

    def test_get_compression_model_default(self):
        """Modèle de compression par défaut."""
        model = self.compressor._get_compression_model()
        assert model == "gemini-3-flash-preview"

    def test_get_compression_model_custom(self):
        """Modèle de compression personnalisé."""
        config = {
            "context_strategy": {
                "compression": {
                    "compression_model": "custom-model",
                }
            }
        }
        compressor = SemanticCompressor(config=config)
        assert compressor._get_compression_model() == "custom-model"
