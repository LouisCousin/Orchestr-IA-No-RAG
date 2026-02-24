"""Acquisition du corpus documentaire depuis fichiers locaux et URLs distantes."""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from src.utils.file_utils import ensure_dir, format_sequence_name, get_next_sequence_number, sanitize_filename

logger = logging.getLogger("orchestria")


@dataclass
class AcquisitionStatus:
    """Statut d'acquisition d'un document."""
    source: str  # URL ou chemin du fichier original
    status: str  # "SUCCESS", "FAILED", "ERROR"
    destination: Optional[str] = None  # Chemin dans le corpus
    message: str = ""
    content_type: Optional[str] = None
    file_size: int = 0


@dataclass
class AcquisitionReport:
    """Rapport d'acquisition du corpus."""
    statuses: list[AcquisitionStatus] = field(default_factory=list)
    total_files: int = 0
    successful: int = 0
    failed: int = 0

    def add(self, status: AcquisitionStatus) -> None:
        self.statuses.append(status)
        self.total_files += 1
        if status.status == "SUCCESS":
            self.successful += 1
        else:
            self.failed += 1


class CorpusAcquirer:
    """Module d'acquisition du corpus documentaire."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md", ".html", ".htm"}

    def __init__(
        self,
        corpus_dir: Path,
        connection_timeout: int = 15,
        read_timeout: int = 60,
        throttle_delay: float = 1.0,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ):
        self.corpus_dir = ensure_dir(corpus_dir)
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
        self.throttle_delay = throttle_delay
        self.user_agent = user_agent
        self._session = None

    def _get_session(self):
        """Crée ou retourne la session HTTP persistante."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            })
        return self._session

    def acquire_local_files(self, file_paths: list[Path], report: Optional[AcquisitionReport] = None) -> AcquisitionReport:
        """Copie des fichiers locaux dans le dossier corpus."""
        if report is None:
            report = AcquisitionReport()

        for path in file_paths:
            path = Path(path)
            try:
                if not path.exists():
                    report.add(AcquisitionStatus(
                        source=str(path), status="FAILED",
                        message=f"Fichier introuvable : {path}"
                    ))
                    continue

                if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    report.add(AcquisitionStatus(
                        source=str(path), status="FAILED",
                        message=f"Format non supporté : {path.suffix}"
                    ))
                    continue

                seq_num = get_next_sequence_number(self.corpus_dir)
                dest_name = format_sequence_name(seq_num, path.stem, path.suffix.lower())
                dest_path = self.corpus_dir / dest_name
                shutil.copy2(path, dest_path)

                report.add(AcquisitionStatus(
                    source=str(path), status="SUCCESS",
                    destination=str(dest_path),
                    message=f"Copié : {dest_name}",
                    file_size=dest_path.stat().st_size,
                ))
                logger.info(f"Fichier local acquis : {path.name} → {dest_name}")

            except Exception as e:
                report.add(AcquisitionStatus(
                    source=str(path), status="ERROR",
                    message=f"Erreur : {e}"
                ))
                logger.error(f"Erreur acquisition fichier {path}: {e}")

        return report

    def acquire_urls(self, urls: list[str], report: Optional[AcquisitionReport] = None) -> AcquisitionReport:
        """Télécharge des documents depuis une liste d'URLs."""
        import requests

        if report is None:
            report = AcquisitionReport()

        for i, url in enumerate(urls):
            url = url.strip()
            if not url:
                continue

            try:
                result = self._download_from_url(url)
                report.add(result)
            except Exception as e:
                report.add(AcquisitionStatus(
                    source=url, status="ERROR",
                    message=f"Erreur inattendue : {e}"
                ))
                logger.error(f"Erreur acquisition URL {url}: {e}")

            # Throttling entre téléchargements
            if i < len(urls) - 1 and self.throttle_delay > 0:
                time.sleep(self.throttle_delay)

        return report

    def _download_from_url(self, url: str) -> AcquisitionStatus:
        """Stratégie de téléchargement en cascade pour une URL."""
        import requests
        from bs4 import BeautifulSoup

        session = self._get_session()
        timeout = (self.connection_timeout, self.read_timeout)
        parsed = urlparse(url)
        domain = sanitize_filename(parsed.netloc or "unknown")

        # Étape 1 : Détection de lien PDF direct
        try:
            head_resp = session.head(url, timeout=timeout, allow_redirects=True)
            content_type = head_resp.headers.get("Content-Type", "").lower()
        except Exception:
            content_type = ""

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            return self._download_file(url, domain, ".pdf", session, timeout)

        # Étape 2 : Télécharger la page et chercher des liens PDF
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            return AcquisitionStatus(source=url, status="FAILED", message=f"Impossible de charger la page : {e}")

        actual_content_type = resp.headers.get("Content-Type", "").lower()

        # Si c'est un PDF malgré tout
        if "application/pdf" in actual_content_type:
            seq_num = get_next_sequence_number(self.corpus_dir)
            dest_name = format_sequence_name(seq_num, domain, ".pdf")
            dest_path = self.corpus_dir / dest_name
            dest_path.write_bytes(resp.content)
            return AcquisitionStatus(
                source=url, status="SUCCESS", destination=str(dest_path),
                message=f"PDF téléchargé : {dest_name}",
                content_type="application/pdf", file_size=len(resp.content),
            )

        # Chercher des liens PDF dans la page HTML
        if "text/html" in actual_content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.lower().endswith(".pdf"):
                    # Résoudre l'URL relative
                    from urllib.parse import urljoin
                    pdf_url = urljoin(url, href)
                    result = self._download_file(pdf_url, domain, ".pdf", session, timeout)
                    if result.status == "SUCCESS":
                        return result

            # Étape 3 : Extraire le contenu textuel de la page HTML
            return self._save_html_as_text(url, resp.text, domain)

        return AcquisitionStatus(
            source=url, status="FAILED",
            message=f"Type de contenu non supporté : {actual_content_type}"
        )

    def _download_file(self, url: str, domain: str, ext: str, session, timeout) -> AcquisitionStatus:
        """Télécharge un fichier depuis une URL."""
        from src.utils.content_validator import is_valid_pdf_content, is_antibot_page

        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()

            # Vérifier que le contenu PDF est bien un PDF (magic bytes)
            if ext == ".pdf" and not is_valid_pdf_content(resp.content):
                # Le contenu n'est pas un PDF, c'est probablement une page HTML de challenge
                try:
                    html_text = resp.content.decode("utf-8", errors="ignore")
                    if is_antibot_page(html_text, url):
                        return AcquisitionStatus(
                            source=url, status="FAILED",
                            message=(
                                "Page de protection anti-bot détectée au lieu du PDF attendu. "
                                "Essayez de télécharger le document manuellement."
                            ),
                        )
                except Exception:
                    pass
                return AcquisitionStatus(
                    source=url, status="FAILED",
                    message="Le contenu téléchargé n'est pas un PDF valide.",
                )

            seq_num = get_next_sequence_number(self.corpus_dir)
            dest_name = format_sequence_name(seq_num, domain, ext)
            dest_path = self.corpus_dir / dest_name
            dest_path.write_bytes(resp.content)

            logger.info(f"Fichier téléchargé : {url} → {dest_name}")
            return AcquisitionStatus(
                source=url, status="SUCCESS", destination=str(dest_path),
                message=f"Téléchargé : {dest_name}",
                content_type=resp.headers.get("Content-Type", ""),
                file_size=len(resp.content),
            )
        except Exception as e:
            return AcquisitionStatus(source=url, status="FAILED", message=f"Échec téléchargement : {e}")

    def _save_html_as_text(self, url: str, html_content: str, domain: str) -> AcquisitionStatus:
        """Sauvegarde le contenu textuel d'une page HTML dans le corpus."""
        from src.core.text_extractor import extract_html
        from src.utils.content_validator import is_antibot_page

        result = extract_html(html_content)
        if result.status == "success" and result.text.strip():
            # Vérifier si c'est une page anti-bot
            if is_antibot_page(result.text, url):
                logger.warning(f"Page anti-bot détectée : {url}")
                return AcquisitionStatus(
                    source=url, status="FAILED",
                    message=(
                        "Page de protection anti-bot détectée (Anubis/Cloudflare). "
                        "Le contenu réel n'est pas accessible. Essayez de télécharger "
                        "le document manuellement et de l'ajouter en fichier local."
                    ),
                )

            seq_num = get_next_sequence_number(self.corpus_dir)
            dest_name = format_sequence_name(seq_num, domain, ".txt")
            dest_path = self.corpus_dir / dest_name
            dest_path.write_text(result.text, encoding="utf-8")

            logger.info(f"Page web extraite : {url} → {dest_name}")
            return AcquisitionStatus(
                source=url, status="SUCCESS", destination=str(dest_path),
                message=f"Contenu web extrait : {dest_name}",
                content_type="text/html", file_size=len(result.text),
            )

        return AcquisitionStatus(
            source=url, status="FAILED",
            message="Aucun contenu textuel extrait de la page"
        )

    def acquire_urls_from_file(self, file_path: Path, report: Optional[AcquisitionReport] = None) -> AcquisitionReport:
        """Charge une liste d'URLs depuis un fichier Excel/CSV."""
        import pandas as pd

        ext = file_path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Chercher la première colonne contenant des URLs
        urls = []
        for col in df.columns:
            values = df[col].dropna().astype(str).tolist()
            if any(v.startswith("http") for v in values):
                urls = [v for v in values if v.startswith("http")]
                break

        if not urls:
            if report is None:
                report = AcquisitionReport()
            report.add(AcquisitionStatus(
                source=str(file_path), status="FAILED",
                message="Aucune URL trouvée dans le fichier"
            ))
            return report

        return self.acquire_urls(urls, report)

    def close(self):
        """Ferme la session HTTP."""
        if self._session:
            self._session.close()
            self._session = None
