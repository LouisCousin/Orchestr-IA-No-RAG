"""Tests unitaires pour le module text_extractor."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.text_extractor import (
    extract,
    extract_pdf,
    extract_docx,
    extract_html,
    extract_text_file,
    extract_excel,
    ExtractionResult,
    _detect_pdf_libraries,
    _AVAILABLE_PDF_LIBS,
    _get_file_hash,
    _save_to_cache,
    _load_from_cache,
    _cache_path_for,
    clear_cache,
    CACHE_DIR,
)


@pytest.fixture
def tmp_text_file(tmp_path):
    """Crée un fichier texte temporaire."""
    f = tmp_path / "test.txt"
    f.write_text("Ceci est un texte de test.\nDeuxième ligne.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_md_file(tmp_path):
    """Crée un fichier Markdown temporaire."""
    f = tmp_path / "test.md"
    f.write_text("# Titre\n\n## Sous-titre\n\nContenu du document.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_html_file(tmp_path):
    """Crée un fichier HTML temporaire."""
    f = tmp_path / "test.html"
    f.write_text(
        "<html><head><title>Test</title><script>var x=1;</script></head>"
        "<body><nav>Menu</nav><h1>Titre</h1><p>Contenu important.</p>"
        "<footer>Pied de page</footer></body></html>",
        encoding="utf-8",
    )
    return f


class TestExtractTextFile:
    def test_extract_txt(self, tmp_text_file):
        result = extract(tmp_text_file)
        assert result.status == "success"
        assert "texte de test" in result.text
        assert result.extraction_method == "direct"
        assert result.char_count > 0
        assert result.word_count > 0
        assert result.hash_text != ""
        assert result.hash_binary != ""

    def test_extract_md(self, tmp_md_file):
        result = extract(tmp_md_file)
        assert result.status == "success"
        assert "Titre" in result.text
        assert result.extraction_method == "direct"

    def test_extract_nonexistent_file(self, tmp_path):
        result = extract(tmp_path / "inexistant.txt")
        assert result.status == "failed"

    def test_extract_unsupported_format(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("contenu")
        result = extract(f)
        assert result.status == "failed"
        assert "non supporté" in result.error_message


# B42: check if optional dependencies are available
try:
    import bs4 as _bs4  # noqa: F401
    _has_bs4 = True
except ImportError:
    _has_bs4 = False


@pytest.mark.skipif(not _has_bs4, reason="beautifulsoup4 not installed")
class TestExtractHTML:
    def test_extract_html_file(self, tmp_html_file):
        result = extract(tmp_html_file)
        assert result.status == "success"
        assert "Contenu important" in result.text
        # Les éléments de navigation doivent être supprimés
        assert "Menu" not in result.text
        assert "var x=1" not in result.text

    def test_extract_html_string(self):
        html = "<html><body><h1>Titre</h1><p>Paragraphe</p><script>code</script></body></html>"
        result = extract_html(html)
        assert result.status == "success"
        assert "Titre" in result.text
        assert "Paragraphe" in result.text
        assert "code" not in result.text

    def test_extract_html_empty(self):
        result = extract_html("<html><body></body></html>")
        assert result.text == "" or result.status == "failed"


class TestExtractPDF:
    @patch("src.core.text_extractor._detect_pdf_libraries")
    def test_no_pdf_library(self, mock_detect, tmp_path):
        mock_detect.return_value = []
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        result = extract_pdf(f)
        assert result.status == "failed"
        assert "aucune" in result.error_message.lower() or result.extraction_method == "none"

    @patch("src.core.text_extractor._detect_pdf_libraries")
    @patch("src.core.text_extractor._PDF_EXTRACTORS", new_callable=dict)
    def test_pymupdf_success(self, mock_extractors, mock_detect, tmp_path):
        mock_detect.return_value = ["pymupdf"]
        mock_extractors["pymupdf"] = MagicMock(return_value=("Contenu PDF extrait", 3))
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        result = extract_pdf(f)
        assert result.status == "success"
        assert result.text == "Contenu PDF extrait"
        assert result.page_count == 3
        assert result.extraction_method == "pymupdf"

    @patch("src.core.text_extractor._detect_pdf_libraries")
    @patch("src.core.text_extractor._PDF_EXTRACTORS", new_callable=dict)
    def test_fallback_to_pdfplumber(self, mock_extractors, mock_detect, tmp_path):
        mock_detect.return_value = ["pymupdf", "pdfplumber"]
        mock_extractors["pymupdf"] = MagicMock(side_effect=Exception("pymupdf error"))
        mock_extractors["pdfplumber"] = MagicMock(return_value=("Contenu pdfplumber", 2))
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        result = extract_pdf(f)
        assert result.status == "success"
        assert result.extraction_method == "pdfplumber"


class TestExtractionResult:
    def test_result_fields(self, tmp_text_file):
        result = extract(tmp_text_file)
        assert result.source_filename == "test.txt"
        assert result.source_size_bytes > 0
        assert result.char_count == len(result.text)
        assert result.word_count == len(result.text.split())

    def test_hash_consistency(self, tmp_text_file):
        result1 = extract(tmp_text_file)
        result2 = extract(tmp_text_file)
        assert result1.hash_binary == result2.hash_binary
        assert result1.hash_text == result2.hash_text


class TestExtractLatin1:
    def test_latin1_fallback(self, tmp_path):
        f = tmp_path / "latin1.txt"
        f.write_bytes("Texte avec accents : é à ü ö".encode("latin-1"))
        result = extract_text_file(f)
        assert result.status == "success"
        assert "accents" in result.text


class TestFileHash:
    def test_hash_deterministic(self, tmp_text_file):
        h1 = _get_file_hash(tmp_text_file)
        h2 = _get_file_hash(tmp_text_file)
        assert h1 == h2

    def test_hash_changes_on_content_change(self, tmp_path):
        f = tmp_path / "changing.txt"
        f.write_text("contenu original")
        h1 = _get_file_hash(f)
        f.write_text("contenu modifié")
        h2 = _get_file_hash(f)
        assert h1 != h2


class TestDiskCache:
    @pytest.fixture(autouse=True)
    def _use_tmp_cache(self, tmp_path, monkeypatch):
        """Redirige CACHE_DIR vers un répertoire temporaire pour chaque test."""
        monkeypatch.setattr("src.core.text_extractor.CACHE_DIR", tmp_path / "cache")

    def test_cache_miss_then_hit(self, tmp_text_file):
        """Premier appel = extraction réelle, deuxième = cache."""
        result1 = extract(tmp_text_file)
        assert result1.status == "success"
        assert "texte de test" in result1.text

        # Le fichier cache doit exister
        file_hash = _get_file_hash(tmp_text_file)
        assert _cache_path_for(file_hash).exists()

        # Deuxième appel : chargé depuis le cache
        result2 = extract(tmp_text_file)
        assert result2.status == "success"
        assert result2.text == result1.text
        assert result2.extraction_method == result1.extraction_method

    def test_cache_force_bypass(self, tmp_text_file):
        """force=True ignore le cache et ré-extrait."""
        result1 = extract(tmp_text_file)
        file_hash = _get_file_hash(tmp_text_file)
        assert _cache_path_for(file_hash).exists()

        result2 = extract(tmp_text_file, force=True)
        assert result2.status == "success"
        assert result2.text == result1.text

    def test_clear_cache(self, tmp_text_file):
        """clear_cache supprime le fichier JSON du cache."""
        extract(tmp_text_file)
        file_hash = _get_file_hash(tmp_text_file)
        cache_file = _cache_path_for(file_hash)
        assert cache_file.exists()

        removed = clear_cache(tmp_text_file)
        assert removed is True
        assert not cache_file.exists()

    def test_clear_cache_absent(self, tmp_text_file):
        """clear_cache retourne False si aucun cache n'existe."""
        removed = clear_cache(tmp_text_file)
        assert removed is False

    def test_cache_invalidated_on_file_change(self, tmp_path):
        """Si le fichier change, le cache est invalidé naturellement."""
        f = tmp_path / "evolving.txt"
        f.write_text("version 1", encoding="utf-8")

        result1 = extract(f)
        hash1 = _get_file_hash(f)
        assert _cache_path_for(hash1).exists()

        f.write_text("version 2 avec plus de contenu", encoding="utf-8")
        hash2 = _get_file_hash(f)
        assert hash1 != hash2
        # L'ancien cache existe encore, mais le nouveau hash n'a pas de cache
        assert not _cache_path_for(hash2).exists()

        result2 = extract(f)
        assert result2.text == "version 2 avec plus de contenu"
        assert _cache_path_for(hash2).exists()

    def test_cache_json_round_trip(self, tmp_text_file):
        """Le JSON du cache contient toutes les propriétés d'ExtractionResult."""
        result = extract(tmp_text_file)
        file_hash = _get_file_hash(tmp_text_file)
        cache_file = _cache_path_for(file_hash)

        with open(cache_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["source_filename"] == "test.txt"
        assert data["status"] == "success"
        assert data["extraction_method"] == "direct"
        assert "texte de test" in data["text"]
