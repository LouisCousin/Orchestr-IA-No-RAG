"""Extraction de texte multi-format avec chaîne de fallback PDF.

Supporte : PDF (docling → pymupdf → pdfplumber → PyPDF2), DOCX, HTML, TXT/MD, Excel/CSV.
Phase 2.5 : ajout de Docling comme extracteur PDF prioritaire avec structure sémantique.
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
    structure: Optional[list[dict]] = None  # Phase 2.5 : sections détectées
    # Chaque dict : {"text": str, "type": str, "page": int, "level": int}


# --- Détection des bibliothèques disponibles ---

_AVAILABLE_PDF_LIBS: list[str] = []


def _detect_pdf_libraries() -> list[str]:
    """Détecte les bibliothèques PDF disponibles, ordonnées par priorité.

    Phase 2.5 : Docling est prioritaire car il préserve la structure sémantique.
    """
    global _AVAILABLE_PDF_LIBS
    if _AVAILABLE_PDF_LIBS:
        return _AVAILABLE_PDF_LIBS

    libs = []
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
        libs.append("docling")
    except ImportError:
        pass
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


def _detect_heading_level(label: str) -> int:
    """Détecte le niveau hiérarchique d'un élément Docling."""
    label_lower = label.lower() if label else ""
    if "title" in label_lower or "heading" in label_lower:
        # Essayer d'extraire le niveau depuis le label (ex: "section_header_1")
        for ch in reversed(label_lower):
            if ch.isdigit():
                return int(ch)
        return 1
    return 0


def _extract_pdf_docling(path: Path) -> tuple[str, int, list[dict]]:
    """Extraction PDF via Docling avec structure sémantique.

    Returns:
        Tuple (texte_complet, nombre_pages, structure_sémantique).
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    sections = []
    for item in doc.iterate_items():
        text = item.text if hasattr(item, "text") else str(item)
        if not text or not text.strip():
            continue
        label = item.label if hasattr(item, "label") else "paragraph"
        page_no = None
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no if hasattr(item.prov[0], "page_no") else None
        sections.append({
            "text": text,
            "type": str(label),
            "page": page_no,
            "level": _detect_heading_level(str(label)),
        })

    full_text = "\n".join(s["text"] for s in sections)
    page_count = doc.num_pages() if hasattr(doc, "num_pages") else max(
        (s.get("page") or 0 for s in sections), default=1
    )

    # Extract title from first heading/title element
    title = None
    for s in sections:
        if s.get("level", 0) >= 1:
            title = s["text"].strip()
            break

    return full_text, page_count, sections, title


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
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages), page_count


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
    """Extrait le texte d'un PDF avec chaîne de fallback.

    Phase 2.5 : Docling est tenté en priorité pour obtenir la structure sémantique.
    """
    available = _detect_pdf_libraries()
    if not available:
        return _make_result(
            path, text="", page_count=0, method="none",
            status="failed", error="Aucune bibliothèque PDF disponible"
        )

    for lib_name in available:
        try:
            if lib_name == "docling":
                # Docling retourne aussi la structure sémantique
                text, page_count, structure, title = _extract_pdf_docling(path)
                if text.strip():
                    logger.info(f"PDF extrait avec docling: {path.name} ({page_count} pages, {len(structure)} éléments)")
                    result = _make_result(path, text=text, page_count=page_count, method="docling", status="success")
                    result.structure = structure
                    if title:
                        result.metadata["title"] = title
                    return result
                else:
                    logger.warning(f"Extraction vide avec docling pour {path.name}, tentative suivante...")
            else:
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
    """Extrait le texte d'un fichier DOCX avec structure sémantique (Phase 2.5)."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = []
        structure = []
        title = None
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
                # Détecter le type et le niveau depuis le style
                style_name = para.style.name if para.style else "Normal"
                if style_name.startswith("Heading"):
                    try:
                        level = int(style_name.split()[-1])
                    except (ValueError, IndexError):
                        level = 1
                    structure.append({
                        "text": para.text,
                        "type": "title",
                        "page": None,
                        "level": level,
                    })
                    if title is None:
                        title = para.text.strip()
                elif style_name == "Title":
                    structure.append({
                        "text": para.text,
                        "type": "title",
                        "page": None,
                        "level": 1,
                    })
                    if title is None:
                        title = para.text.strip()
                else:
                    structure.append({
                        "text": para.text,
                        "type": "paragraph",
                        "page": None,
                        "level": 0,
                    })
        text = "\n\n".join(paragraphs)
        page_count = max(1, len(text) // 3000)  # Estimation
        result = _make_result(path, text=text, page_count=page_count, method="python-docx", status="success")
        if structure:
            result.structure = structure
        if title:
            result.metadata["title"] = title
        return result
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

        # Extract title from <title> tag or first <h1>
        title = None
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
        if not title:
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.get_text(strip=True)

        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        page_count = max(1, len(text) // 3000)

        if source_path:
            result = _make_result(source_path, text=text, page_count=page_count, method="beautifulsoup", status="success")
            if title:
                result.metadata["title"] = title
            return result
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
