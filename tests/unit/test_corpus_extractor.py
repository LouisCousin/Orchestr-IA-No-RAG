"""Tests unitaires pour le module corpus_extractor."""

import pytest

from src.core.corpus_extractor import CorpusChunk, StructuredCorpus


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def small_corpus():
    """Corpus avec 3 documents (palier full_excerpts)."""
    chunks = [
        CorpusChunk(text="Doc A intro " * 50, source_file="alpha.pdf", chunk_index=0, char_count=650),
        CorpusChunk(text="Doc A suite " * 50, source_file="alpha.pdf", chunk_index=1, char_count=650),
        CorpusChunk(text="Doc B intro " * 50, source_file="beta.docx", chunk_index=0, char_count=650),
        CorpusChunk(text="Doc B suite " * 50, source_file="beta.docx", chunk_index=1, char_count=650),
        CorpusChunk(text="Doc C intro " * 50, source_file="gamma.txt", chunk_index=0, char_count=650),
    ]
    return StructuredCorpus(
        chunks=chunks, total_chunks=len(chunks),
        source_files=["alpha.pdf", "beta.docx", "gamma.txt"],
    )


@pytest.fixture
def medium_corpus():
    """Corpus avec 25 documents (palier first_sentences)."""
    chunks = []
    source_files = []
    for i in range(25):
        name = f"doc_{i:02d}.pdf"
        source_files.append(name)
        chunks.append(CorpusChunk(
            text=f"Première phrase du document {i}. Suite du texte qui est plus longue.",
            source_file=name, chunk_index=0, char_count=60,
        ))
    return StructuredCorpus(
        chunks=chunks, total_chunks=len(chunks), source_files=source_files,
    )


@pytest.fixture
def large_corpus():
    """Corpus avec 100 documents (palier sampled)."""
    chunks = []
    source_files = []
    for i in range(100):
        name = f"document_{i:03d}.pdf"
        source_files.append(name)
        chunks.append(CorpusChunk(
            text=f"Contenu du document numéro {i}. " * 20,
            source_file=name, chunk_index=0, char_count=600,
        ))
    return StructuredCorpus(
        chunks=chunks, total_chunks=len(chunks), source_files=source_files,
    )


@pytest.fixture
def single_doc_corpus():
    """Corpus avec un seul document."""
    chunks = [
        CorpusChunk(text="Contenu unique.", source_file="seul.pdf", chunk_index=0, char_count=15),
    ]
    return StructuredCorpus(chunks=chunks, total_chunks=1, source_files=["seul.pdf"])


@pytest.fixture
def empty_corpus():
    return StructuredCorpus()


# ── Tests palier full_excerpts (1-10 docs) ──────────────────────────

class TestDigestFullExcerpts:
    def test_tier_is_full_excerpts(self, small_corpus):
        digest = small_corpus.get_corpus_digest()
        assert digest["tier"] == "full_excerpts"

    def test_returns_one_entry_per_document(self, small_corpus):
        digest = small_corpus.get_corpus_digest()
        source_files = [e["source_file"] for e in digest["entries"]]
        assert source_files == ["alpha.pdf", "beta.docx", "gamma.txt"]

    def test_entries_have_text(self, small_corpus):
        digest = small_corpus.get_corpus_digest()
        for entry in digest["entries"]:
            assert len(entry["text"]) > 0

    def test_respects_total_budget(self, small_corpus):
        budget = 300
        digest = small_corpus.get_corpus_digest(max_total_chars=budget)
        total_chars = sum(len(e["text"]) for e in digest["entries"])
        assert total_chars <= budget

    def test_equal_distribution(self, small_corpus):
        budget = 300
        digest = small_corpus.get_corpus_digest(max_total_chars=budget)
        lengths = [len(e["text"]) for e in digest["entries"]]
        for length in lengths:
            assert length <= budget // 3

    def test_uses_beginning_of_documents(self, small_corpus):
        digest = small_corpus.get_corpus_digest(max_total_chars=600)
        assert digest["entries"][0]["text"].startswith("Doc A intro")
        assert digest["entries"][1]["text"].startswith("Doc B intro")
        assert digest["entries"][2]["text"].startswith("Doc C intro")

    def test_single_document(self, single_doc_corpus):
        digest = single_doc_corpus.get_corpus_digest()
        assert digest["tier"] == "full_excerpts"
        assert digest["num_documents"] == 1
        assert len(digest["entries"]) == 1
        assert "Contenu unique" in digest["entries"][0]["text"]

    def test_num_documents_matches(self, small_corpus):
        digest = small_corpus.get_corpus_digest()
        assert digest["num_documents"] == 3


# ── Tests palier first_sentences (11-50 docs) ──────────────────────

class TestDigestFirstSentences:
    def test_tier_is_first_sentences(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        assert digest["tier"] == "first_sentences"

    def test_num_documents(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        assert digest["num_documents"] == 25

    def test_one_entry_per_document(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        assert len(digest["entries"]) == 25

    def test_entries_contain_first_sentence(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        for entry in digest["entries"]:
            # Each text should be a short sentence, not the full doc
            assert len(entry["text"]) <= 200
            assert len(entry["text"]) > 0

    def test_text_is_first_sentence_not_full_text(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        entry = digest["entries"][0]
        # Should be just the first sentence, ending with period
        assert entry["text"].endswith(".")
        assert "Suite du texte" not in entry["text"]


# ── Tests palier sampled (51+ docs) ────────────────────────────────

class TestDigestSampled:
    def test_tier_is_sampled(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        assert digest["tier"] == "sampled"

    def test_num_documents(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        assert digest["num_documents"] == 100

    def test_sample_size_is_limited(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        assert len(digest["entries"]) <= 5

    def test_all_filenames_present(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        assert "all_filenames" in digest
        assert len(digest["all_filenames"]) == 100

    def test_sampled_entries_have_text(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        for entry in digest["entries"]:
            assert len(entry["text"]) > 0

    def test_sampled_entries_are_spread_out(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        filenames = [e["source_file"] for e in digest["entries"]]
        # With 100 docs, step=20, should sample docs 0, 20, 40, 60, 80
        # They should not all be from the beginning
        indices = [int(f.split("_")[1].split(".")[0]) for f in filenames]
        assert max(indices) - min(indices) > 10  # spread across the corpus


# ── Tests empty corpus ─────────────────────────────────────────────

class TestDigestEmpty:
    def test_empty_corpus_returns_empty(self, empty_corpus):
        digest = empty_corpus.get_corpus_digest()
        assert digest["tier"] == "full_excerpts"
        assert digest["num_documents"] == 0
        assert digest["entries"] == []


# ── Tests boundary: exactly at tier thresholds ─────────────────────

class TestDigestBoundaries:
    def test_exactly_10_docs_uses_full_excerpts(self):
        chunks = [
            CorpusChunk(text=f"Text for doc {i}.", source_file=f"d{i}.pdf", chunk_index=0)
            for i in range(10)
        ]
        corpus = StructuredCorpus(chunks=chunks, total_chunks=10, source_files=[f"d{i}.pdf" for i in range(10)])
        assert corpus.get_corpus_digest()["tier"] == "full_excerpts"

    def test_exactly_11_docs_uses_first_sentences(self):
        chunks = [
            CorpusChunk(text=f"First sentence of doc {i}. More text follows.", source_file=f"d{i}.pdf", chunk_index=0)
            for i in range(11)
        ]
        corpus = StructuredCorpus(chunks=chunks, total_chunks=11, source_files=[f"d{i}.pdf" for i in range(11)])
        assert corpus.get_corpus_digest()["tier"] == "first_sentences"

    def test_exactly_50_docs_uses_first_sentences(self):
        chunks = [
            CorpusChunk(text=f"Sentence for {i}. More.", source_file=f"d{i}.pdf", chunk_index=0)
            for i in range(50)
        ]
        corpus = StructuredCorpus(chunks=chunks, total_chunks=50, source_files=[f"d{i}.pdf" for i in range(50)])
        assert corpus.get_corpus_digest()["tier"] == "first_sentences"

    def test_exactly_51_docs_uses_sampled(self):
        chunks = [
            CorpusChunk(text=f"Content {i}.", source_file=f"d{i}.pdf", chunk_index=0)
            for i in range(51)
        ]
        corpus = StructuredCorpus(chunks=chunks, total_chunks=51, source_files=[f"d{i}.pdf" for i in range(51)])
        assert corpus.get_corpus_digest()["tier"] == "sampled"
