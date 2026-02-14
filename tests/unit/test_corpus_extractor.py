"""Tests unitaires pour le module corpus_extractor."""

import pytest

from src.core.corpus_extractor import (
    CorpusChunk,
    StructuredCorpus,
    extract_keywords_tfidf,
)


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
    """Corpus avec 25 documents (palier first_sentences), textes distincts."""
    topics = [
        "cloud computing infrastructure serveur hébergement",
        "intelligence artificielle apprentissage automatique neural",
        "cybersécurité protection données intrusion firewall",
        "blockchain cryptomonnaie décentralisation ledger",
        "développement logiciel agile sprint déploiement",
    ]
    chunks = []
    source_files = []
    for i in range(25):
        name = f"doc_{i:02d}.pdf"
        source_files.append(name)
        topic = topics[i % len(topics)]
        chunks.append(CorpusChunk(
            text=f"Première phrase du document {i}. {topic} " * 10,
            source_file=name, chunk_index=0, char_count=500,
        ))
    return StructuredCorpus(
        chunks=chunks, total_chunks=len(chunks), source_files=source_files,
    )


@pytest.fixture
def large_corpus():
    """Corpus avec 100 documents (palier sampled), textes distincts."""
    topics = [
        "marché financier bourse investissement rendement",
        "énergie renouvelable solaire éolien transition",
        "santé publique épidémiologie vaccin prévention",
        "transport logistique chaîne approvisionnement livraison",
        "éducation pédagogie formation enseignement numérique",
    ]
    chunks = []
    source_files = []
    for i in range(100):
        name = f"document_{i:03d}.pdf"
        source_files.append(name)
        topic = topics[i % len(topics)]
        chunks.append(CorpusChunk(
            text=f"Contenu du document numéro {i}. {topic} " * 15,
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


# ── Tests extract_keywords_tfidf ─────────────────────────────────────

class TestExtractKeywordsTfidf:
    def test_basic_extraction(self):
        docs = {
            "cloud.pdf": "Le cloud computing permet le stockage cloud et le calcul cloud distribué.",
            "ia.pdf": "L'intelligence artificielle et le machine learning transforment la santé.",
        }
        kw = extract_keywords_tfidf(docs, top_k=3)
        assert "cloud.pdf" in kw
        assert "ia.pdf" in kw
        assert "cloud" in kw["cloud.pdf"]
        # "intelligence" or "artificielle" or "machine" or "learning" should be in ia.pdf
        assert any(w in kw["ia.pdf"] for w in ("intelligence", "artificielle", "machine", "learning", "santé"))

    def test_distinctive_words_rank_higher(self):
        docs = {
            "a.txt": "python python python développement web",
            "b.txt": "java java java développement mobile",
        }
        kw = extract_keywords_tfidf(docs, top_k=2)
        assert "python" in kw["a.txt"]
        assert "java" in kw["b.txt"]
        # "développement" is common to both, should not rank as high
        assert "développement" not in kw["a.txt"][:1]

    def test_stopwords_excluded(self):
        docs = {"a.txt": "le la les de des un une est dans pour avec"}
        kw = extract_keywords_tfidf(docs, top_k=5)
        assert kw["a.txt"] == []

    def test_empty_docs(self):
        assert extract_keywords_tfidf({}) == {}

    def test_single_document_returns_keywords(self):
        docs = {"only.pdf": "algorithme optimisation algorithme performance algorithme"}
        kw = extract_keywords_tfidf(docs, top_k=3)
        # With 1 doc, IDF = log(1/1) = 0 for all words -> no keywords with score > 0
        # This is expected: TF-IDF needs >1 doc to discriminate
        assert "only.pdf" in kw

    def test_top_k_limit(self):
        docs = {
            "a.txt": "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
            "b.txt": "kilo lima mike november oscar papa quebec romeo sierra tango",
        }
        kw = extract_keywords_tfidf(docs, top_k=3)
        assert len(kw["a.txt"]) <= 3
        assert len(kw["b.txt"]) <= 3

    def test_short_words_excluded(self):
        docs = {"a.txt": "AI ML DB OS IT VR AR XR le la"}
        kw = extract_keywords_tfidf(docs, top_k=5)
        # All words are < 3 chars, should be empty
        assert kw["a.txt"] == []


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

    def test_no_keywords_in_full_excerpts(self, small_corpus):
        digest = small_corpus.get_corpus_digest()
        for entry in digest["entries"]:
            assert "keywords" not in entry

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
            assert len(entry["text"]) <= 200
            assert len(entry["text"]) > 0

    def test_text_is_first_sentence_not_full_text(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        entry = digest["entries"][0]
        assert entry["text"].endswith(".")

    def test_entries_have_keywords(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        entries_with_kw = [e for e in digest["entries"] if e.get("keywords")]
        # Most entries should have keywords (some might not if text is too generic)
        assert len(entries_with_kw) > 0

    def test_keywords_are_lists_of_strings(self, medium_corpus):
        digest = medium_corpus.get_corpus_digest()
        for entry in digest["entries"]:
            if "keywords" in entry:
                assert isinstance(entry["keywords"], list)
                for kw in entry["keywords"]:
                    assert isinstance(kw, str)


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
        indices = [int(f.split("_")[1].split(".")[0]) for f in filenames]
        assert max(indices) - min(indices) > 10

    def test_all_files_keywords_present(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        assert "all_files_keywords" in digest
        assert len(digest["all_files_keywords"]) == 100

    def test_all_files_keywords_structure(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        for fkw in digest["all_files_keywords"]:
            assert "source_file" in fkw
            assert "keywords" in fkw
            assert isinstance(fkw["keywords"], list)

    def test_keywords_cover_all_documents(self, large_corpus):
        digest = large_corpus.get_corpus_digest()
        kw_files = {fkw["source_file"] for fkw in digest["all_files_keywords"]}
        all_files = set(digest["all_filenames"])
        assert kw_files == all_files


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
