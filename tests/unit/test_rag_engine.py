"""Tests unitaires pour le moteur RAG."""

import pytest
from unittest.mock import patch, MagicMock

from src.core.rag_engine import RAGEngine, RAGResult


class TestRAGResult:
    """Tests de la dataclass RAGResult."""

    def test_default_values(self):
        result = RAGResult(section_id="1", section_title="Test")
        assert result.chunks == []
        assert result.scores == []
        assert result.avg_score == 0.0
        assert result.num_relevant == 0
        assert result.total_tokens == 0

    def test_to_dict(self):
        result = RAGResult(
            section_id="1.1",
            section_title="Introduction",
            avg_score=0.75,
            num_relevant=5,
            total_tokens=1000,
        )
        d = result.to_dict()
        assert d["section_id"] == "1.1"
        assert d["avg_score"] == 0.75
        assert d["num_relevant"] == 5


class TestRAGEngineTextSplitting:
    """Tests du découpage de texte."""

    def test_short_text_no_split(self):
        engine = RAGEngine(chunk_size=100, chunk_overlap=10)
        chunks = engine._split_text("Court texte.")
        assert len(chunks) == 1
        assert chunks[0] == "Court texte."

    def test_empty_text(self):
        engine = RAGEngine(chunk_size=100, chunk_overlap=10)
        chunks = engine._split_text("")
        assert chunks == []

    def test_long_text_splits(self):
        engine = RAGEngine(chunk_size=50, chunk_overlap=10)
        text = "Mot " * 200  # Long text
        chunks = engine._split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0


class TestRAGEngineIndexing:
    """Tests de l'indexation du corpus (avec mock ChromaDB)."""

    @patch("src.core.rag_engine.RAGEngine._get_collection")
    def test_index_corpus(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection

        engine = RAGEngine(chunk_size=100, chunk_overlap=10)
        count = engine.index_corpus([
            {"text": "Document un avec du contenu.", "source_file": "doc1.txt"},
            {"text": "Document deux avec autre contenu.", "source_file": "doc2.txt"},
        ])

        assert count == 2
        mock_collection.add.assert_called_once()

    @patch("src.core.rag_engine.RAGEngine._get_collection")
    def test_index_empty_corpus(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection

        engine = RAGEngine()
        count = engine.index_corpus([])
        assert count == 0

    @patch("src.core.rag_engine.RAGEngine._get_collection")
    def test_index_skip_empty_text(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection

        engine = RAGEngine(chunk_size=100, chunk_overlap=10)
        count = engine.index_corpus([
            {"text": "", "source_file": "empty.txt"},
            {"text": "   ", "source_file": "whitespace.txt"},
        ])
        assert count == 0


class TestRAGEngineSearch:
    """Tests de la recherche RAG."""

    @patch("src.core.rag_engine.RAGEngine._get_collection")
    def test_search_empty_collection(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_get_collection.return_value = mock_collection

        engine = RAGEngine()
        result = engine.search("test query")
        assert result.chunks == []
        assert result.avg_score == 0.0

    @patch("src.core.rag_engine.RAGEngine._get_collection")
    def test_search_returns_results(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["Texte pertinent A", "Texte pertinent B"]],
            "metadatas": [[
                {"source_file": "doc1.txt", "chunk_index": 0, "token_estimate": 10},
                {"source_file": "doc2.txt", "chunk_index": 1, "token_estimate": 15},
            ]],
            "distances": [[0.2, 0.6]],
        }
        mock_get_collection.return_value = mock_collection

        engine = RAGEngine(relevance_threshold=0.3)
        result = engine.search("query", top_k=5)

        assert len(result.chunks) == 2
        assert result.chunks[0]["source_file"] == "doc1.txt"
        # Distance 0.2 → similarity 0.9
        assert result.chunks[0]["similarity"] == 0.9
        assert result.num_relevant == 2
        assert result.total_tokens == 25

    def test_search_for_section(self):
        engine = RAGEngine()
        engine.search = MagicMock(return_value=RAGResult(
            section_id="", section_title="",
            chunks=[{"text": "chunk"}], scores=[0.8],
            avg_score=0.8, num_relevant=1,
        ))
        result = engine.search_for_section("1.1", "Introduction", "Description")
        assert result.section_id == "1.1"
        assert result.section_title == "Introduction"


class TestRAGEngineProperties:
    """Tests des propriétés."""

    @patch("src.core.rag_engine.RAGEngine._get_collection")
    def test_indexed_count(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_get_collection.return_value = mock_collection

        engine = RAGEngine()
        assert engine.indexed_count == 42
