"""Tests unitaires pour le module corpus_acquirer."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.corpus_acquirer import CorpusAcquirer, AcquisitionReport, AcquisitionStatus

# B42: check if bs4 is available for HTML-related tests
try:
    import bs4 as _bs4  # noqa: F401
    _has_bs4 = True
except ImportError:
    _has_bs4 = False


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    return d


@pytest.fixture
def acquirer(corpus_dir):
    return CorpusAcquirer(corpus_dir=corpus_dir)


class TestAcquireLocalFiles:
    def test_copy_txt_file(self, acquirer, corpus_dir, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("Contenu du fichier")
        report = acquirer.acquire_local_files([src])
        assert report.successful == 1
        assert report.failed == 0
        files = list(corpus_dir.glob("001_*"))
        assert len(files) == 1

    def test_copy_multiple_files(self, acquirer, corpus_dir, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"Contenu {i}")
            files.append(f)
        report = acquirer.acquire_local_files(files)
        assert report.successful == 3
        corpus_files = list(corpus_dir.glob("*"))
        assert len(corpus_files) == 3

    def test_nonexistent_file(self, acquirer):
        report = acquirer.acquire_local_files([Path("/nonexistent/file.txt")])
        assert report.failed == 1
        assert report.successful == 0

    def test_unsupported_format(self, acquirer, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"fake png")
        report = acquirer.acquire_local_files([f])
        assert report.failed == 1

    def test_sequential_naming(self, acquirer, corpus_dir, tmp_path):
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content {i}")
            acquirer.acquire_local_files([f])

        files = sorted(corpus_dir.glob("*.txt"))
        assert files[0].name.startswith("001_")
        assert files[1].name.startswith("002_")
        assert files[2].name.startswith("003_")


class TestAcquireURLs:
    @patch("src.core.corpus_acquirer.CorpusAcquirer._get_session")
    def test_download_pdf_direct(self, mock_session_fn, acquirer, corpus_dir):
        session = MagicMock()
        mock_session_fn.return_value = session

        # HEAD retourne application/pdf
        head_resp = MagicMock()
        head_resp.headers = {"Content-Type": "application/pdf"}
        session.head.return_value = head_resp

        # GET retourne le contenu PDF
        get_resp = MagicMock()
        get_resp.content = b"%PDF-1.4 fake pdf content"
        get_resp.headers = {"Content-Type": "application/pdf"}
        get_resp.raise_for_status = MagicMock()
        session.get.return_value = get_resp

        report = acquirer.acquire_urls(["https://example.com/doc.pdf"])
        assert report.successful == 1

    @pytest.mark.skipif(not _has_bs4, reason="beautifulsoup4 not installed")
    @patch("src.core.corpus_acquirer.CorpusAcquirer._get_session")
    def test_download_html_with_pdf_link(self, mock_session_fn, acquirer, corpus_dir):
        session = MagicMock()
        mock_session_fn.return_value = session

        # HEAD retourne text/html
        head_resp = MagicMock()
        head_resp.headers = {"Content-Type": "text/html"}
        session.head.return_value = head_resp

        # GET pour la page HTML
        html_resp = MagicMock()
        html_resp.headers = {"Content-Type": "text/html"}
        html_resp.text = '<html><body><a href="/documents/report.pdf">PDF</a></body></html>'
        html_resp.raise_for_status = MagicMock()
        html_resp.status_code = 200

        # GET pour le PDF trouvé dans la page
        pdf_resp = MagicMock()
        pdf_resp.content = b"%PDF-1.4 content"
        pdf_resp.headers = {"Content-Type": "application/pdf"}
        pdf_resp.raise_for_status = MagicMock()

        session.get.side_effect = [html_resp, pdf_resp]

        report = acquirer.acquire_urls(["https://example.com/page"])
        assert report.successful == 1

    @pytest.mark.skipif(not _has_bs4, reason="beautifulsoup4 not installed")
    @patch("src.core.corpus_acquirer.CorpusAcquirer._get_session")
    def test_download_html_extract_text(self, mock_session_fn, acquirer, corpus_dir):
        session = MagicMock()
        mock_session_fn.return_value = session

        head_resp = MagicMock()
        head_resp.headers = {"Content-Type": "text/html"}
        session.head.return_value = head_resp

        html_resp = MagicMock()
        html_resp.headers = {"Content-Type": "text/html"}
        html_resp.text = '<html><body><h1>Article</h1><p>Contenu textuel important.</p></body></html>'
        html_resp.raise_for_status = MagicMock()
        html_resp.status_code = 200
        session.get.return_value = html_resp

        report = acquirer.acquire_urls(["https://example.com/article"])
        assert report.successful == 1
        txt_files = list(corpus_dir.glob("*.txt"))
        assert len(txt_files) == 1

    def test_empty_url_skipped(self, acquirer):
        report = acquirer.acquire_urls(["", "  "])
        assert report.total_files == 0


class TestAcquisitionReport:
    def test_report_counters(self):
        report = AcquisitionReport()
        report.add(AcquisitionStatus(source="a.pdf", status="SUCCESS", message="ok"))
        report.add(AcquisitionStatus(source="b.pdf", status="FAILED", message="err"))
        report.add(AcquisitionStatus(source="c.pdf", status="SUCCESS", message="ok"))
        assert report.total_files == 3
        assert report.successful == 2
        assert report.failed == 1
