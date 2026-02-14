"""Tests unitaires pour le module text_extractor."""

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
