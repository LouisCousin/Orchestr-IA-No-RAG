"""Extraction de texte multi-format avec chaîne de fallback PDF.

Supporte : PDF (pymupdf → pdfplumber → PyPDF2), DOCX, HTML, TXT/MD, Excel/CSV.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.file_utils import sha256_file, sha256_text

logger = logging.getLogger("orchestria")


@dataclass
class ExtractionResult:
    """Résultat d'extraction de texte d'un document."""
    text: str
    page_count: int
    char_count: int
    word_count: int
    extraction_method: str
    status: str  # "success", "partial", "failed"
    source_filename: str
    source_size_bytes: int
    source_mime_type: Optional[str] = None
    source_modified: Optional[str] = None
    hash_binary: str = ""
    hash_text: str = ""
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# --- Détection des bibliothèques disponibles ---

_AVAILABLE_PDF_LIBS: list[str] = []


def _detect_pdf_libraries() -> list[str]:
    """Détecte les bibliothèques PDF disponibles, ordonnées par priorité."""
    global _AVAILABLE_PDF_LIBS
    if _AVAILABLE_PDF_LIBS:
        return _AVAILABLE_PDF_LIBS

    libs = []
    try:
        import fitz  # noqa: F401
        libs.append("pymupdf")
    except ImportError:
        pass
    try:
        import pdfplumber  # noqa: F401
        libs.append("pdfplumber")
    except ImportError:
        pass
    try:
        import PyPDF2  # noqa: F401
        libs.append("PyPDF2")
    except ImportError:
        pass

    _AVAILABLE_PDF_LIBS = libs
    logger.info(f"Bibliothèques PDF disponibles : {libs if libs else 'aucune'}")
    return libs


# --- Extraction PDF par bibliothèque ---

def _extract_pdf_pymupdf(path: Path) -> tuple[str, int]:
    """Extraction PDF via pymupdf (fitz)."""
    import fitz
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    page_count = len(pages)
    doc.close()
    return "\n\n".join(pages), page_count


def _extract_pdf_pdfplumber(path: Path) -> tuple[str, int]:
    """Extraction PDF via pdfplumber."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages), len(pdf.pages)


def _extract_pdf_pypdf2(path: Path) -> tuple[str, int]:
    """Extraction PDF via PyPDF2."""
    import PyPDF2
    pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages), len(reader.pages)


_PDF_EXTRACTORS = {
    "pymupdf": _extract_pdf_pymupdf,
    "pdfplumber": _extract_pdf_pdfplumber,
    "PyPDF2": _extract_pdf_pypdf2,
}


def extract_pdf(path: Path) -> ExtractionResult:
    """Extrait le texte d'un PDF avec chaîne de fallback."""
    available = _detect_pdf_libraries()
    if not available:
        return _make_result(
            path, text="", page_count=0, method="none",
            status="failed", error="Aucune bibliothèque PDF disponible"
        )

    for lib_name in available:
        try:
            extractor = _PDF_EXTRACTORS[lib_name]
            text, page_count = extractor(path)
            if text.strip():
                logger.info(f"PDF extrait avec {lib_name}: {path.name} ({page_count} pages)")
                return _make_result(path, text=text, page_count=page_count, method=lib_name, status="success")
            else:
                logger.warning(f"Extraction vide avec {lib_name} pour {path.name}, tentative suivante...")
        except Exception as e:
            logger.warning(f"Échec extraction PDF avec {lib_name} pour {path.name}: {e}")
            continue

    return _make_result(
        path, text="", page_count=0, method="all_failed",
        status="failed", error="Toutes les bibliothèques d'extraction ont échoué"
    )


# --- Extraction DOCX ---

def extract_docx(path: Path) -> ExtractionResult:
    """Extrait le texte d'un fichier DOCX."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        text = "\n\n".join(paragraphs)
        page_count = max(1, len(text) // 3000)  # Estimation
        return _make_result(path, text=text, page_count=page_count, method="python-docx", status="success")
    except Exception as e:
        return _make_result(path, text="", page_count=0, method="python-docx", status="failed", error=str(e))


# --- Extraction HTML ---

def extract_html(path_or_text: Path | str) -> ExtractionResult:
    """Extrait le texte d'un fichier HTML en supprimant les éléments non textuels."""
    try:
        from bs4 import BeautifulSoup

        if isinstance(path_or_text, Path):
            html_content = path_or_text.read_text(encoding="utf-8", errors="replace")
            source_path = path_or_text
        else:
            html_content = path_or_text
            source_path = None

        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        page_count = max(1, len(text) // 3000)

        if source_path:
            return _make_result(source_path, text=text, page_count=page_count, method="beautifulsoup", status="success")
        else:
            return ExtractionResult(
                text=text, page_count=page_count,
                char_count=len(text), word_count=len(text.split()),
                extraction_method="beautifulsoup", status="success",
                source_filename="web_content", source_size_bytes=len(html_content),
                hash_binary="", hash_text=sha256_text(text) if text else "",
            )
    except Exception as e:
        if isinstance(path_or_text, Path):
            return _make_result(path_or_text, text="", page_count=0, method="beautifulsoup", status="failed", error=str(e))
        return ExtractionResult(
            text="", page_count=0, char_count=0, word_count=0,
            extraction_method="beautifulsoup", status="failed",
            source_filename="web_content", source_size_bytes=0,
            error_message=str(e),
        )


# --- Extraction Excel/CSV ---

def extract_excel(path: Path) -> ExtractionResult:
    """Extrait le texte d'un fichier Excel ou CSV."""
    try:
        import pandas as pd
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
            text = df.to_string(index=False)
        else:
            dfs = pd.read_excel(path, sheet_name=None)
            parts = []
            for sheet_name, df in dfs.items():
                parts.append(f"=== Feuille: {sheet_name} ===\n{df.to_string(index=False)}")
            text = "\n\n".join(parts)
        page_count = max(1, len(text) // 3000)
        return _make_result(path, text=text, page_count=page_count, method="pandas", status="success")
    except Exception as e:
        return _make_result(path, text="", page_count=0, method="pandas", status="failed", error=str(e))


# --- Extraction TXT/Markdown ---

def extract_text_file(path: Path) -> ExtractionResult:
    """Lit un fichier texte brut ou Markdown."""
    try:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        page_count = max(1, len(text) // 3000)
        return _make_result(path, text=text, page_count=page_count, method="direct", status="success")
    except Exception as e:
        return _make_result(path, text="", page_count=0, method="direct", status="failed", error=str(e))


# --- Routeur principal ---

def extract(path: Path) -> ExtractionResult:
    """Extrait le texte d'un fichier selon son extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".docx":
        return extract_docx(path)
    elif ext in (".html", ".htm"):
        return extract_html(path)
    elif ext in (".xlsx", ".xls", ".csv"):
        return extract_excel(path)
    elif ext in (".txt", ".md", ".markdown"):
        return extract_text_file(path)
    else:
        return _make_result(
            path, text="", page_count=0, method="unsupported",
            status="failed", error=f"Format non supporté : {ext}"
        )


# --- Helpers ---

def _make_result(
    path: Path,
    text: str,
    page_count: int,
    method: str,
    status: str,
    error: Optional[str] = None,
) -> ExtractionResult:
    """Construit un ExtractionResult avec les métadonnées du fichier."""
    try:
        stat = path.stat()
        size = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
    except OSError:
        size = 0
        modified = None

    return ExtractionResult(
        text=text,
        page_count=page_count,
        char_count=len(text),
        word_count=len(text.split()) if text else 0,
        extraction_method=method,
        status=status,
        source_filename=path.name,
        source_size_bytes=size,
        source_modified=modified,
        hash_binary=sha256_file(path) if path.exists() and status != "failed" else "",
        hash_text=sha256_text(text) if text else "",
        error_message=error,
    )
