"""Tests unitaires pour grobid_client.py."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.core.grobid_client import GrobidClient


SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Test Article Title</title>
      </titleStmt>
      <publicationStmt>
        <date when="2024"/>
      </publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <surname>Dupont</surname>
                <forename>Jean</forename>
              </persName>
            </author>
            <author>
              <persName>
                <surname>Smith</surname>
                <forename>Alice</forename>
              </persName>
            </author>
          </analytic>
          <monogr>
            <title level="j">Journal of Testing</title>
            <imprint>
              <biblScope unit="volume">42</biblScope>
              <biblScope unit="issue">3</biblScope>
              <biblScope unit="page" from="100" to="120"/>
              <publisher>Test Publisher</publisher>
            </imprint>
          </monogr>
          <idno type="DOI">10.1234/test.2024.001</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>This is the abstract.</abstract>
    </profileDesc>
  </teiHeader>
</TEI>"""


@pytest.fixture
def client():
    return GrobidClient(enabled=True, server_url="http://localhost:8070")


@pytest.fixture
def disabled_client():
    return GrobidClient(enabled=False)


class TestGrobidClient:
    def test_disabled_not_available(self, disabled_client):
        assert disabled_client.is_available() is False

    def test_disabled_process_header_returns_empty(self, disabled_client, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        assert disabled_client.process_header(pdf) == {}

    @patch("src.core.grobid_client.requests.get")
    def test_is_available_success(self, mock_get, client):
        mock_get.return_value = MagicMock(status_code=200)
        assert client.is_available() is True

    @patch("src.core.grobid_client.requests.get")
    def test_is_available_failure(self, mock_get, client):
        mock_get.side_effect = Exception("Connection refused")
        assert client.is_available() is False

    def test_parse_tei_header(self, client):
        result = client._parse_tei_header(SAMPLE_TEI)
        assert result["title"] == "Test Article Title"
        assert result["authors"] == ["Dupont, J.", "Smith, A."]
        assert result["year"] == 2024
        assert result["journal"] == "Journal of Testing"
        assert result["volume"] == "42"
        assert result["issue"] == "3"
        assert result["pages"] == "100-120"
        assert result["doi"] == "10.1234/test.2024.001"
        assert result["publisher"] == "Test Publisher"
        assert result["abstract"] == "This is the abstract."

    def test_parse_tei_header_empty(self, client):
        result = client._parse_tei_header("<TEI xmlns='http://www.tei-c.org/ns/1.0'></TEI>")
        assert result == {}

    def test_parse_tei_header_invalid_xml(self, client):
        result = client._parse_tei_header("not xml at all")
        assert result == {}

    def test_extract_year(self):
        assert GrobidClient._extract_year("2024-01-15") == 2024
        assert GrobidClient._extract_year("Published in 1998") == 1998
        assert GrobidClient._extract_year("no year") is None

    @patch("src.core.grobid_client.requests.post")
    def test_process_header_success(self, mock_post, client, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        mock_post.return_value = MagicMock(status_code=200, text=SAMPLE_TEI)
        result = client.process_header(pdf)
        assert result["title"] == "Test Article Title"

    @patch("src.core.grobid_client.requests.post")
    def test_process_header_server_error(self, mock_post, client, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        mock_post.return_value = MagicMock(status_code=500)
        result = client.process_header(pdf)
        assert result == {}

    @patch("src.core.grobid_client.requests.post")
    def test_process_header_connection_error(self, mock_post, client, tmp_path):
        import requests
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        result = client.process_header(pdf)
        assert result == {}

    def test_process_header_missing_file(self, client):
        result = client.process_header(Path("/nonexistent/file.pdf"))
        assert result == {}

    @patch("src.core.grobid_client.requests.post")
    def test_process_batch(self, mock_post, client, tmp_path):
        pdf1 = tmp_path / "test1.pdf"
        pdf2 = tmp_path / "test2.pdf"
        pdf1.write_bytes(b"pdf1")
        pdf2.write_bytes(b"pdf2")
        mock_post.return_value = MagicMock(status_code=200, text=SAMPLE_TEI)
        results = client.process_batch([pdf1, pdf2])
        assert len(results) == 2
