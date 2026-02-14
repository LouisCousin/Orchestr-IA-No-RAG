"""Tests unitaires pour le module corpus_extractor."""

import pytest

from src.core.corpus_extractor import CorpusChunk, StructuredCorpus


@pytest.fixture
def multi_doc_corpus():
    """Corpus avec 3 documents sources, plusieurs chunks chacun."""
    chunks = [
        CorpusChunk(text="Doc A intro " * 50, source_file="alpha.pdf", chunk_index=0, char_count=650),
        CorpusChunk(text="Doc A suite " * 50, source_file="alpha.pdf", chunk_index=1, char_count=650),
        CorpusChunk(text="Doc B intro " * 50, source_file="beta.docx", chunk_index=0, char_count=650),
        CorpusChunk(text="Doc B suite " * 50, source_file="beta.docx", chunk_index=1, char_count=650),
        CorpusChunk(text="Doc C intro " * 50, source_file="gamma.txt", chunk_index=0, char_count=650),
    ]
    corpus = StructuredCorpus(
        chunks=chunks,
        total_chunks=len(chunks),
        source_files=["alpha.pdf", "beta.docx", "gamma.txt"],
    )
    return corpus


@pytest.fixture
def single_doc_corpus():
    """Corpus avec un seul document."""
    chunks = [
        CorpusChunk(text="Contenu unique.", source_file="seul.pdf", chunk_index=0, char_count=15),
    ]
    return StructuredCorpus(
        chunks=chunks, total_chunks=1, source_files=["seul.pdf"],
    )


@pytest.fixture
def empty_corpus():
    return StructuredCorpus()


class TestGetCorpusDigest:
    def test_returns_one_entry_per_document(self, multi_doc_corpus):
        digest = multi_doc_corpus.get_corpus_digest()
        source_files = [d["source_file"] for d in digest]
        assert source_files == ["alpha.pdf", "beta.docx", "gamma.txt"]

    def test_excerpts_are_not_empty(self, multi_doc_corpus):
        digest = multi_doc_corpus.get_corpus_digest()
        for entry in digest:
            assert len(entry["excerpt"]) > 0

    def test_respects_total_budget(self, multi_doc_corpus):
        budget = 300
        digest = multi_doc_corpus.get_corpus_digest(max_total_chars=budget)
        total_chars = sum(len(d["excerpt"]) for d in digest)
        assert total_chars <= budget

    def test_equal_distribution(self, multi_doc_corpus):
        budget = 300
        digest = multi_doc_corpus.get_corpus_digest(max_total_chars=budget)
        lengths = [len(d["excerpt"]) for d in digest]
        # Each doc should get roughly budget/3 = 100 chars
        for length in lengths:
            assert length <= budget // 3

    def test_single_document(self, single_doc_corpus):
        digest = single_doc_corpus.get_corpus_digest()
        assert len(digest) == 1
        assert digest[0]["source_file"] == "seul.pdf"
        assert "Contenu unique" in digest[0]["excerpt"]

    def test_empty_corpus(self, empty_corpus):
        digest = empty_corpus.get_corpus_digest()
        assert digest == []

    def test_default_budget_is_reasonable(self, multi_doc_corpus):
        digest = multi_doc_corpus.get_corpus_digest()
        total_chars = sum(len(d["excerpt"]) for d in digest)
        # Default budget is 8000 chars (~2000 tokens)
        assert total_chars <= 8000

    def test_uses_beginning_of_documents(self, multi_doc_corpus):
        digest = multi_doc_corpus.get_corpus_digest(max_total_chars=600)
        # First chunk text starts with "Doc A intro", "Doc B intro", "Doc C intro"
        assert digest[0]["excerpt"].startswith("Doc A intro")
        assert digest[1]["excerpt"].startswith("Doc B intro")
        assert digest[2]["excerpt"].startswith("Doc C intro")
