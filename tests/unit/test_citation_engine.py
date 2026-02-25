"""Tests unitaires pour citation_engine.py."""

import pytest
from src.core.citation_engine import CitationEngine, CitationRef, BibliographyEntry


class TestFormatAPA:
    def test_article(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="article",
            authors="Dupont, J.",
            year=2024,
            title="Mon article",
            journal="Revue Test",
            volume="45",
            issue="2",
            pages_range="123-145",
            doi="10.1234/test",
        )
        assert "Dupont, J." in ref
        assert "(2024)" in ref
        assert "*Revue Test*" in ref
        assert "https://doi.org/10.1234/test" in ref

    def test_book(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="book",
            authors="Smith, A.",
            year=2023,
            title="Mon livre",
            publisher="Éditions Test",
        )
        assert "*Mon livre*" in ref
        assert "Éditions Test" in ref

    def test_chapter(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="chapter",
            authors="Martin, B.",
            year=2022,
            title="Mon chapitre",
            editor="Dupont, J.",
            book_title="Le livre collectif",
            pages_range="50-75",
            publisher="Éditeur",
        )
        assert "Mon chapitre" in ref
        assert "Dans Dupont, J." in ref
        assert "(p. 50-75)" in ref

    def test_report(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="report",
            authors="Organisation X",
            year=2024,
            title="Rapport annuel",
        )
        assert "*Rapport annuel*" in ref

    def test_web(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="web",
            authors="Blog Author",
            year=2024,
            title="Article en ligne",
            site_name="Le Blog",
            url="https://example.com",
        )
        assert "*Le Blog*" in ref
        assert "https://example.com" in ref

    def test_thesis(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="thesis",
            authors="Étudiant, E.",
            year=2024,
            title="Ma thèse",
            university="Université de Test",
        )
        assert "[Thèse de doctorat, Université de Test]" in ref

    def test_no_author(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="article",
            title="Sans auteur",
            year=2024,
        )
        assert "Auteur inconnu" in ref

    def test_no_year(self):
        ref = CitationEngine.format_apa_reference(
            doc_type="article",
            authors="Dupont, J.",
            title="Article sans date",
        )
        assert "(s.d.)" in ref


class TestExtractCitations:
    def test_simple_citation(self):
        text = "Selon (Dupont, 2024), ceci est vrai."
        citations = CitationEngine.extract_inline_citations(text)
        assert len(citations) == 1
        assert citations[0].authors == "Dupont"
        assert citations[0].year == 2024

    def test_et_al_citation(self):
        text = "(Dupont et al., 2023)"
        citations = CitationEngine.extract_inline_citations(text)
        assert len(citations) == 1
        assert "et al." in citations[0].authors

    def test_multiple_citations(self):
        text = "(Dupont, 2024) et aussi (Smith, 2023)"
        citations = CitationEngine.extract_inline_citations(text)
        assert len(citations) == 2

    def test_no_citations(self):
        text = "Aucune citation ici."
        citations = CitationEngine.extract_inline_citations(text)
        assert len(citations) == 0

    def test_accented_name(self):
        text = "(Émond, 2024)"
        citations = CitationEngine.extract_inline_citations(text)
        assert len(citations) == 1


class TestResolveCitations:
    def test_resolve_without_store(self):
        engine = CitationEngine(metadata_store=None, enabled=True)
        citations = [CitationRef("(Dupont, 2024)", "Dupont", 2024)]
        result = engine.resolve_citations(citations)
        assert result[0].resolved_doc_id is None

    def test_resolve_disabled(self):
        engine = CitationEngine(enabled=False)
        citations = [CitationRef("(Dupont, 2024)", "Dupont", 2024)]
        result = engine.resolve_citations(citations)
        assert result[0].resolved_doc_id is None


class TestCompileBibliography:
    def test_empty_bibliography(self):
        engine = CitationEngine(enabled=True)
        entries = engine.compile_bibliography()
        assert entries == []

    def test_disabled_returns_empty(self):
        engine = CitationEngine(enabled=False)
        assert engine.compile_bibliography() == []


class TestFormatFromMetadata:
    def test_with_dict(self):
        doc = {
            "doc_type": "article",
            "authors": '["Dupont, J."]',
            "year": 2024,
            "title": "Test article",
            "journal": "Revue",
        }
        ref = CitationEngine.format_apa_from_metadata(doc)
        assert "Dupont" in ref
        assert "(2024)" in ref

    def test_with_list_authors(self):
        doc = {
            "doc_type": "book",
            "authors": ["Smith, A.", "Jones, B."],
            "year": 2023,
            "title": "Test book",
        }
        ref = CitationEngine.format_apa_from_metadata(doc)
        assert "Smith" in ref


class TestCitationEngineState:
    def test_get_cited_doc_ids(self):
        engine = CitationEngine(enabled=True)
        engine._cited_doc_ids.add("doc1")
        engine._cited_doc_ids.add("doc2")
        assert len(engine.get_cited_doc_ids()) == 2

    def test_reset(self):
        engine = CitationEngine(enabled=True)
        engine._cited_doc_ids.add("doc1")
        engine.reset()
        assert len(engine.get_cited_doc_ids()) == 0
