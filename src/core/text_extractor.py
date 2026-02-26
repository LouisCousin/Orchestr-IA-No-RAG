"""Extraction de texte multi-format avec chaîne de fallback PDF.

Supporte : PDF (docling → pymupdf → pdfplumber → PyPDF2), DOCX, HTML, TXT/MD, Excel/CSV.
Phase 2.5 : ajout de Docling comme extracteur PDF prioritaire avec structure sémantique.
Phase 4 (Perf) : parallélisation ProcessPoolExecutor + psutil pour gestion dynamique des ressources,
                  cache par hash pour éviter les ré-extractions, singleton DocumentConverter par worker.
                  compute_optimal_workers() est publique pour réutilisation par d'autres modules.
"""

import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.config import load_default_config
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


def _load_pdf_extraction_config() -> dict:
    """Charge la configuration pdf_extraction depuis default.yaml avec valeurs par défaut."""
    try:
        cfg = load_default_config()
        pdf_cfg = cfg.get("pdf_extraction", {}) or {}
    except Exception:
        pdf_cfg = {}
    return {
        "docling_page_batch_size": pdf_cfg.get("docling_page_batch_size", 30),
        "docling_batch_threshold": pdf_cfg.get("docling_batch_threshold", 50),
        "coverage_threshold": pdf_cfg.get("coverage_threshold", 0.80),
        "disable_page_images": pdf_cfg.get("disable_page_images", True),
        "disable_picture_classification": pdf_cfg.get("disable_picture_classification", True),
        "disable_ocr": pdf_cfg.get("disable_ocr", True),
    }


def _create_docling_converter(pdf_cfg: dict):
    """Crée une instance DocumentConverter avec les options de pipeline optimisées."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = not pdf_cfg.get("disable_ocr", True)
    pipeline_options.generate_page_images = not pdf_cfg["disable_page_images"]
    pipeline_options.generate_picture_images = not pdf_cfg.get("disable_picture_images", True)
    pipeline_options.do_picture_classification = not pdf_cfg["disable_picture_classification"]

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )
    return converter


def _extract_sections_from_docling_result(result) -> list[dict]:
    """Extrait les sections structurées depuis un résultat Docling."""
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
    return sections


# --- Gestion dynamique des ressources (Smart Resources) ---


def compute_optimal_workers() -> int:
    """Calcule le nombre optimal de workers selon RAM et CPU disponibles.

    Formule : workers = min(CPU_COUNT, (RAM_DISPO_GB - 2GB_SECURITY) // 1.5GB_PER_WORKER)
    Fallback : toujours au moins 1 worker.

    Cette fonction est publique pour être réutilisée par d'autres modules
    (ex: corpus_acquirer pour la parallélisation de l'extraction).
    """
    try:
        import psutil
        ram_available_gb = psutil.virtual_memory().available / (1024 ** 3)
        cpu_count = os.cpu_count() or 1
        ram_workers = int((ram_available_gb - 2.0) / 1.5)
        workers = max(1, min(cpu_count, ram_workers))
        logger.info(
            f"Smart Resources : RAM dispo={ram_available_gb:.1f}GB, "
            f"CPU={cpu_count}, workers calculés={workers}"
        )
        return workers
    except ImportError:
        logger.warning("psutil non disponible, fallback à 1 worker")
        return 1
    except Exception as e:
        logger.warning(f"Erreur calcul workers : {e}, fallback à 1")
        return 1


# --- Worker Multiprocessing pour extraction Docling parallèle ---

# Variables globales du worker (initialisées une seule fois par processus)
_worker_converter = None
_worker_pdf_cfg = None


def _init_docling_worker(pdf_cfg: dict) -> None:
    """Initializer pour ProcessPoolExecutor : charge DocumentConverter une seule fois par worker."""
    global _worker_converter, _worker_pdf_cfg
    _worker_pdf_cfg = pdf_cfg
    _worker_converter = _create_docling_converter(pdf_cfg)
    logger.info("Worker Docling initialisé (DocumentConverter chargé)")


def _docling_worker_extract_batch(args: tuple) -> list[dict]:
    """Fonction worker : extrait une plage de pages depuis un PDF.

    Args:
        args: Tuple (path_str, start_page, end_page).

    Returns:
        Liste de sections extraites (dicts).
    """
    global _worker_converter
    path_str, start, end = args
    _logger = logging.getLogger("orchestria")

    try:
        from docling.datamodel.settings import PageRange
        path = Path(path_str)
        page_range: PageRange = (start, end)
        result = _worker_converter.convert(str(path), page_range=page_range)
        sections = _extract_sections_from_docling_result(result)
        _logger.info(f"  Worker lot pages {start}-{end} → {len(sections)} éléments")
        del result
        gc.collect()
        return sections
    except Exception as e:
        _logger.warning(f"  Worker échec lot pages {start}-{end}: {e}")
        return []


# --- Cache d'extraction par hash ---


def is_extraction_cached(path: Path, metadata_store) -> bool:
    """Vérifie si un fichier a déjà été extrait avec succès via son hash binaire.

    Args:
        path: Chemin du fichier à vérifier.
        metadata_store: Instance de MetadataStore.

    Returns:
        True si l'extraction est déjà complète dans le store.
    """
    if metadata_store is None:
        return False
    try:
        file_hash = sha256_file(path)
        if not file_hash:
            return False
        conn = metadata_store._get_conn()
        row = conn.execute(
            "SELECT extraction_status FROM documents WHERE hash_binary = ? AND extraction_status = 'success'",
            (file_hash,),
        ).fetchone()
        if row:
            logger.info(f"Cache hit : {path.name} déjà extrait (hash={file_hash[:12]}...)")
            return True
    except Exception as e:
        logger.debug(f"Erreur vérification cache pour {path.name}: {e}")
    return False


def _get_pdf_page_count(path: Path) -> int:
    """Obtient le nombre de pages d'un PDF via pypdfium2 ou fitz."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        count = len(pdf)
        pdf.close()
        return count
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(str(path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        pass
    return 0


def _extract_pdf_docling(path: Path) -> tuple[str, int, list[dict], str | None, str, str]:
    """Extraction PDF via Docling avec structure sémantique.

    Inclut :
    - Options de pipeline optimisées (pas d'images, backend PyPdfium2)
    - Traitement parallèle par lots via ProcessPoolExecutor pour les gros PDF (>50 pages)
    - Singleton DocumentConverter par worker (chargé une seule fois)
    - Gestion dynamique des workers via psutil (RAM/CPU)
    - Détection de couverture et rattrapage pymupdf pour les pages manquantes

    Returns:
        Tuple (texte_complet, nombre_pages, structure_sémantique, titre, méthode, statut).
    """
    pdf_cfg = _load_pdf_extraction_config()
    batch_threshold = pdf_cfg["docling_batch_threshold"]
    batch_size = pdf_cfg["docling_page_batch_size"]
    coverage_threshold = pdf_cfg["coverage_threshold"]

    # Déterminer le nombre total de pages
    total_pages = _get_pdf_page_count(path)
    logger.info(f"PDF {path.name} : {total_pages} pages détectées, seuil batch = {batch_threshold}")
    if total_pages == 0:
        logger.warning(f"Impossible de déterminer le nombre de pages de {path.name}, conversion en une passe")

    sections = []
    method = "docling"

    if total_pages > batch_threshold:
        # ── Mode batch parallèle : ProcessPoolExecutor avec singleton Docling ──
        max_workers = compute_optimal_workers()

        # Construire les plages de pages
        page_ranges = []
        for start in range(1, total_pages + 1, batch_size):
            end = min(start + batch_size - 1, total_pages)
            page_ranges.append((str(path), start, end))

        logger.info(
            f"PDF volumineux ({total_pages} pages), {len(page_ranges)} lots, "
            f"{max_workers} workers parallèles"
        )

        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_docling_worker,
                initargs=(pdf_cfg,),
            ) as executor:
                futures = {
                    executor.submit(_docling_worker_extract_batch, args): args
                    for args in page_ranges
                }
                for future in as_completed(futures):
                    args = futures[future]
                    try:
                        batch_sections = future.result()
                        sections.extend(batch_sections)
                    except Exception as e:
                        logger.warning(f"  Échec lot pages {args[1]}-{args[2]}: {e}")
        except Exception as e:
            logger.warning(f"Échec ProcessPoolExecutor, fallback séquentiel : {e}")
            # Fallback séquentiel en cas d'erreur du pool
            from docling.datamodel.settings import PageRange
            for path_str, start, end in page_ranges:
                try:
                    converter = _create_docling_converter(pdf_cfg)
                    page_range: PageRange = (start, end)
                    result = converter.convert(str(path), page_range=page_range)
                    batch_sections = _extract_sections_from_docling_result(result)
                    sections.extend(batch_sections)
                except Exception as batch_err:
                    logger.warning(f"  Échec lot séquentiel pages {start}-{end}: {batch_err}")
                finally:
                    try:
                        del converter
                        del result
                    except NameError:
                        pass
                    gc.collect()
    else:
        # ── Mode single-pass : PDF de taille modérée ──
        converter = None
        result = None
        try:
            converter = _create_docling_converter(pdf_cfg)
            result = converter.convert(str(path))
            sections = _extract_sections_from_docling_result(result)

            # Récupérer le page_count depuis Docling si on ne l'a pas
            if total_pages == 0:
                doc = result.document
                total_pages = doc.num_pages() if hasattr(doc, "num_pages") else max(
                    (s.get("page") or 0 for s in sections), default=1
                )
        except Exception as e:
            logger.warning(f"Erreur extraction Docling single-pass pour {path.name}: {e}")
            raise
        finally:
            if converter:
                del converter
            if result:
                del result
            gc.collect()

    # ── Détection de couverture et rattrapage pymupdf ──
    if total_pages > 0:
        pages_covered = set()
        for section in sections:
            if section.get("page"):
                pages_covered.add(section["page"])
        coverage_ratio = len(pages_covered) / total_pages
    else:
        coverage_ratio = 1.0  # pas de référence, on considère OK

    if total_pages > 0 and coverage_ratio < coverage_threshold:
        missing_pages = set(range(1, total_pages + 1)) - pages_covered
        logger.warning(
            f"Docling n'a couvert que {len(pages_covered)}/{total_pages} pages "
            f"({coverage_ratio:.0%}), rattrapage pymupdf pour {len(missing_pages)} pages manquantes"
        )
        try:
            import fitz
            doc = fitz.open(str(path))
            recovered = 0
            for page_num in sorted(missing_pages):
                page = doc[page_num - 1]  # fitz est 0-indexed
                page_text = page.get_text()
                if page_text.strip():
                    sections.append({
                        "text": page_text,
                        "type": "paragraph",
                        "page": page_num,
                        "level": 0,
                    })
                    recovered += 1
            doc.close()
            logger.info(f"Rattrapage pymupdf : {recovered} pages récupérées sur {len(missing_pages)} manquantes")
            method = "docling+pymupdf"
        except ImportError:
            logger.warning("pymupdf non disponible pour le rattrapage des pages manquantes")
        except Exception as e:
            logger.warning(f"Échec du rattrapage pymupdf : {e}")

    # Recalculer la couverture après rattrapage pymupdf
    if method == "docling+pymupdf" and total_pages > 0:
        pages_covered_after = set(s.get("page") for s in sections if s.get("page"))
        coverage_ratio = len(pages_covered_after) / total_pages
        logger.info(f"Couverture après rattrapage : {coverage_ratio:.0%}")

    # ── Reconstruction du résultat ──
    sections.sort(key=lambda s: s.get("page") or 0)
    full_text = "\n".join(s["text"] for s in sections)
    page_count = total_pages if total_pages > 0 else max(
        (s.get("page") or 0 for s in sections), default=1
    )

    # Déterminer le statut
    if coverage_ratio >= coverage_threshold:
        status = "success"
    elif coverage_ratio >= 0.50:
        status = "partial"
    else:
        status = "partial"
        logger.warning(f"Couverture faible même après rattrapage : {coverage_ratio:.0%}")

    # Extraire le titre depuis le premier heading
    title = None
    for s in sections:
        if s.get("level", 0) >= 1:
            title = s["text"].strip()
            break

    return full_text, page_count, sections, title, method, status


def _extract_pdf_metadata(path: Path) -> dict:
    """Extrait les métadonnées (auteur, date) d'un PDF via PyPDF2."""
    metadata = {}
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            info = reader.metadata
            if info:
                if info.author:
                    metadata["author"] = info.author
                if getattr(info, "creation_date", None):
                    metadata["creation_date"] = str(info.creation_date)
    except Exception:
        pass
    return metadata


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
        page_count = len(reader.pages)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages), page_count


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

    # Extraire les métadonnées (auteur, date) indépendamment du texte
    pdf_meta = _extract_pdf_metadata(path)

    for lib_name in available:
        try:
            if lib_name == "docling":
                # Docling retourne aussi la structure sémantique + méthode + statut
                text, page_count, structure, title, method, status = _extract_pdf_docling(path)
                if text.strip():
                    logger.info(
                        f"PDF extrait avec {method}: {path.name} "
                        f"({page_count} pages, {len(structure)} éléments, statut={status})"
                    )
                    result = _make_result(path, text=text, page_count=page_count, method=method, status=status)
                    result.structure = structure
                    if title:
                        result.metadata["title"] = title
                    # Enrichir avec les métadonnées PDF
                    result.metadata.update(pdf_meta)
                    return result
                else:
                    logger.warning(f"Extraction vide avec docling pour {path.name}, tentative suivante...")
            else:
                extractor = _PDF_EXTRACTORS[lib_name]
                text, page_count = extractor(path)
                if text.strip():
                    logger.info(f"PDF extrait avec {lib_name}: {path.name} ({page_count} pages)")
                    result = _make_result(path, text=text, page_count=page_count, method=lib_name, status="success")
                    # Enrichir avec les métadonnées PDF
                    result.metadata.update(pdf_meta)
                    return result
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
        # Extraire auteur depuis les propriétés du document DOCX
        try:
            if doc.core_properties.author:
                result.metadata["author"] = doc.core_properties.author
            if doc.core_properties.created:
                result.metadata["creation_date"] = str(doc.core_properties.created)
        except Exception:
            pass
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

        # Extract author from <meta name="author"> tag
        author = None
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag and author_tag.get("content"):
            author = author_tag["content"].strip()

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
            if author:
                result.metadata["author"] = author
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

def extract(path: Path, metadata_store=None) -> ExtractionResult:
    """Extrait le texte d'un fichier selon son extension.

    Args:
        path: Chemin du fichier à extraire.
        metadata_store: Instance optionnelle de MetadataStore pour vérifier le cache.
            Si le fichier est déjà extrait (hash_binary match, status=success), retourne
            un résultat "cached" sans ré-extraction.
    """
    # Vérification cache avant extraction
    if metadata_store is not None and is_extraction_cached(path, metadata_store):
        return _make_result(
            path, text="", page_count=0, method="cached",
            status="cached",
        )

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
