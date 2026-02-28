"""Client GROBID pour l'extraction de métadonnées bibliographiques.

Phase 3 : communique avec l'API REST de GROBID (conteneur Docker)
pour extraire les métadonnées d'en-tête des documents PDF.
"""

import logging
import re
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger("orchestria")

# Namespace TEI utilisé par GROBID
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _find_text(element, xpath: str, namespaces: Optional[dict] = None) -> Optional[str]:
    """Trouve le texte d'un élément XML, ou None."""
    ns = namespaces or TEI_NS
    el = element.find(xpath, ns)
    if el is not None:
        # Récupère tout le texte, y compris les éléments enfants
        text = "".join(el.itertext()).strip()
        return text if text else None
    return None


class GrobidClient:
    """Client pour l'API REST GROBID."""

    def __init__(
        self,
        server_url: str = "http://localhost:8070",
        timeout_seconds: int = 30,
        consolidate_header: bool = True,
        batch_size: int = 5,
        enabled: bool = False,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout_seconds
        self.consolidate_header = consolidate_header
        self.batch_size = batch_size
        self.enabled = enabled

    def is_available(self) -> bool:
        """Vérifie si le serveur GROBID est accessible."""
        if not self.enabled:
            return False
        try:
            response = requests.get(
                f"{self.server_url}/api/isalive",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def process_header(self, pdf_path: Path) -> dict:
        """Extrait les métadonnées d'en-tête d'un PDF via GROBID.

        Args:
            pdf_path: Chemin vers le fichier PDF.

        Returns:
            Dict avec les métadonnées extraites :
            {title, authors, year, journal, volume, issue, pages, doi, publisher, abstract}
            Retourne un dict vide si GROBID est indisponible ou échoue.
        """
        if not self.enabled:
            return {}

        if not pdf_path.exists():
            logger.warning(f"GROBID : fichier introuvable : {pdf_path}")
            return {}

        url = f"{self.server_url}/api/processHeaderDocument"
        params = {}
        if self.consolidate_header:
            params["consolidateHeader"] = "1"

        for attempt in range(2):  # 1 essai + 1 retry
            try:
                with open(pdf_path, "rb") as f:
                    response = requests.post(
                        url,
                        files={"input": (pdf_path.name, f, "application/pdf")},
                        data=params,
                        timeout=self.timeout,
                    )
                if response.status_code == 200:
                    return self._parse_tei_header(response.text)
                elif response.status_code >= 500 and attempt == 0:
                    logger.warning(
                        f"GROBID : statut {response.status_code} pour {pdf_path.name}, retry..."
                    )
                    continue
                else:
                    logger.warning(
                        f"GROBID : statut {response.status_code} pour {pdf_path.name}"
                    )
                    return {}
            except requests.Timeout:
                if attempt == 0:
                    logger.warning(f"GROBID : timeout pour {pdf_path.name}, retry...")
                    continue
                logger.warning(f"GROBID : timeout après retry pour {pdf_path.name}")
                return {}
            except requests.ConnectionError:
                logger.warning(f"GROBID : serveur inaccessible ({self.server_url})")
                return {}
            except Exception as e:
                logger.warning(f"GROBID : erreur inattendue pour {pdf_path.name}: {e}")
                return {}

        return {}

    def process_batch(self, pdf_paths: list[Path]) -> list[dict]:
        """Traite un lot de PDFs.

        Args:
            pdf_paths: Liste de chemins PDF.

        Returns:
            Liste de dicts de métadonnées (même ordre que pdf_paths).
        """
        results = []
        for path in pdf_paths:
            result = self.process_header(path)
            results.append(result)
        return results

    def _parse_tei_header(self, tei_xml: str) -> dict:
        """Parse la réponse TEI XML de GROBID.

        Returns:
            Dict avec les métadonnées extraites.
        """
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            logger.warning(f"GROBID : erreur parsing TEI XML : {e}")
            return {}

        result = {}

        # Titre
        title = _find_text(root, ".//tei:titleStmt/tei:title")
        if title:
            result["title"] = title

        # Auteurs
        authors = self._extract_authors(root)
        if authors:
            result["authors"] = authors

        # Date / Année
        date_el = root.find(".//tei:publicationStmt/tei:date", TEI_NS)
        if date_el is None:
            date_el = root.find(".//tei:biblStruct//tei:date", TEI_NS)
        if date_el is not None:
            when = date_el.get("when", "")
            year = self._extract_year(when) or self._extract_year(date_el.text or "")
            if year:
                result["year"] = year

        # Journal
        journal = _find_text(root, ".//tei:monogr/tei:title[@level='j']")
        if journal:
            result["journal"] = journal

        # Volume
        vol_el = root.find(".//tei:monogr/tei:imprint/tei:biblScope[@unit='volume']", TEI_NS)
        if vol_el is not None and vol_el.text:
            result["volume"] = vol_el.text.strip()

        # Issue
        issue_el = root.find(".//tei:monogr/tei:imprint/tei:biblScope[@unit='issue']", TEI_NS)
        if issue_el is not None and issue_el.text:
            result["issue"] = issue_el.text.strip()

        # Pages
        page_el = root.find(".//tei:monogr/tei:imprint/tei:biblScope[@unit='page']", TEI_NS)
        if page_el is not None:
            from_page = page_el.get("from", "")
            to_page = page_el.get("to", "")
            if from_page and to_page:
                result["pages"] = f"{from_page}-{to_page}"
            elif page_el.text:
                result["pages"] = page_el.text.strip()

        # DOI
        doi_el = root.find(".//tei:idno[@type='DOI']", TEI_NS)
        if doi_el is not None and doi_el.text:
            result["doi"] = doi_el.text.strip()

        # Publisher
        publisher = _find_text(root, ".//tei:monogr/tei:imprint/tei:publisher")
        if publisher:
            result["publisher"] = publisher

        # Abstract
        abstract = _find_text(root, ".//tei:profileDesc/tei:abstract")
        if abstract:
            result["abstract"] = abstract

        return result

    @staticmethod
    def _extract_authors(root) -> list[str]:
        """Extrait les noms des auteurs depuis le TEI XML."""
        authors = []
        for author_el in root.findall(".//tei:fileDesc//tei:author", TEI_NS):
            persname = author_el.find("tei:persName", TEI_NS)
            if persname is None:
                continue
            surname = _find_text(persname, "tei:surname")
            forename = _find_text(persname, "tei:forename")
            if surname:
                if forename:
                    # Format APA : Nom, Initiale.
                    initial = forename[0].upper() + "."
                    authors.append(f"{surname}, {initial}")
                else:
                    authors.append(surname)
        return authors

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        """Extrait une année (19xx ou 20xx) depuis un texte."""
        match = re.search(r'(19|20)\d{2}', text)
        if match:
            return int(match.group())
        return None
