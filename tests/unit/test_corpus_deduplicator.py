"""Tests unitaires pour le module corpus_deduplicator."""

import pytest
from pathlib import Path

from src.core.corpus_deduplicator import CorpusDeduplicator, DeduplicationReport
from src.core.text_extractor import ExtractionResult


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    return d


@pytest.fixture
def deduplicator(corpus_dir):
    return CorpusDeduplicator(corpus_dir)


def _make_extraction(filename, text, hash_binary=None, hash_text=None):
    """Crée un ExtractionResult de test."""
    from src.utils.file_utils import sha256_text
    import hashlib
    if hash_binary is None:
        hash_binary = hashlib.sha256(text.encode()).hexdigest()
    if hash_text is None:
        hash_text = sha256_text(text)
    return ExtractionResult(
        text=text,
        page_count=1,
        char_count=len(text),
        word_count=len(text.split()),
        extraction_method="test",
        status="success",
        source_filename=filename,
        source_size_bytes=len(text),
        hash_binary=hash_binary,
        hash_text=hash_text,
    )


class TestCheckDuplicate:
    def test_unique_document(self, deduplicator):
        ext = _make_extraction("doc1.txt", "Contenu unique du premier document")
        entry = deduplicator.check_duplicate(ext)
        assert entry.status == "unique"
        assert entry.original_ref is None

    def test_exact_duplicate(self, deduplicator):
        ext1 = _make_extraction("doc1.txt", "Contenu identique")
        deduplicator.register(deduplicator.check_duplicate(ext1))

        ext2 = _make_extraction("doc2.txt", "Contenu identique")
        entry = deduplicator.check_duplicate(ext2)
        assert entry.status == "doublon_exact"
        assert entry.original_ref == "doc1.txt"

    def test_content_duplicate_different_binary(self, deduplicator):
        text = "Contenu textuel identique"
        from src.utils.file_utils import sha256_text

        ext1 = _make_extraction("doc1.pdf", text, hash_binary="hash_binary_1")
        deduplicator.register(deduplicator.check_duplicate(ext1))

        ext2 = _make_extraction("doc2.pdf", text, hash_binary="hash_binary_2")
        entry = deduplicator.check_duplicate(ext2)
        assert entry.status == "doublon_probable"
        assert entry.original_ref == "doc1.pdf"


class TestDeduplicateCorpus:
    def test_corpus_with_duplicates(self, deduplicator):
        extractions = [
            _make_extraction("doc1.txt", "Premier document unique"),
            _make_extraction("doc2.txt", "Deuxième document unique"),
            _make_extraction("doc3.txt", "Premier document unique"),  # Doublon exact
        ]
        report = deduplicator.deduplicate_corpus(extractions)
        assert report.unique_files == 2
        assert report.exact_duplicates == 1
        assert report.content_duplicates == 0

    def test_corpus_all_unique(self, deduplicator):
        extractions = [
            _make_extraction("doc1.txt", "Premier document"),
            _make_extraction("doc2.txt", "Deuxième document"),
            _make_extraction("doc3.txt", "Troisième document"),
        ]
        report = deduplicator.deduplicate_corpus(extractions)
        assert report.unique_files == 3
        assert report.exact_duplicates == 0

    def test_tokens_saved(self, deduplicator):
        text = "Un texte assez long pour avoir des tokens significatifs " * 10
        extractions = [
            _make_extraction("doc1.txt", text),
            _make_extraction("doc2.txt", text),  # Doublon
        ]
        report = deduplicator.deduplicate_corpus(extractions)
        assert report.tokens_saved > 0


class TestSaveLoadIndex:
    def test_save_and_reload(self, corpus_dir):
        dedup1 = CorpusDeduplicator(corpus_dir)
        ext = _make_extraction("doc1.txt", "Contenu test")
        entry = dedup1.check_duplicate(ext)
        dedup1.register(entry)
        report = DeduplicationReport(entries=[entry], unique_files=1)
        dedup1.save_index(report)

        # Recharger
        dedup2 = CorpusDeduplicator(corpus_dir)
        ext_dup = _make_extraction("doc2.txt", "Contenu test")
        entry2 = dedup2.check_duplicate(ext_dup)
        assert entry2.status in ("doublon_exact", "doublon_probable")


class TestRemoveDuplicate:
    def test_remove_existing(self, deduplicator, corpus_dir):
        f = corpus_dir / "001_test.txt"
        f.write_text("contenu")
        assert deduplicator.remove_duplicate("001_test.txt") is True
        assert not f.exists()

    def test_remove_nonexistent(self, deduplicator):
        assert deduplicator.remove_duplicate("inexistant.txt") is False
