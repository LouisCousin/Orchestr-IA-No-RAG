"""Tests d'intégration pour le pipeline de citations APA."""

import pytest
from unittest.mock import MagicMock
from src.core.citation_engine import CitationEngine, CitationRef, BibliographyEntry


class MockDocumentMetadata:
    """Mock d'un document du metadata_store."""
    def __init__(self, doc_id, title, authors, year, journal=None, volume=None,
                 issue=None, pages_range=None, doi=None, publisher=None,
                 doc_type="article", apa_reference=None):
        self.doc_id = doc_id
        self.title = title
        self.authors = authors
        self.year = year
        self.journal = journal
        self.volume = volume
        self.issue = issue
        self.pages_range = pages_range
        self.doi = doi
        self.publisher = publisher
        self.doc_type = doc_type
        self.apa_reference = apa_reference


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_all_documents.return_value = [
        MockDocumentMetadata(
            doc_id="doc_001",
            title="Machine Learning for Education",
            authors='["Dupont, J.", "Smith, A."]',
            year=2024,
            journal="Journal of AI",
            volume="42",
            issue="3",
            pages_range="100-120",
            doi="10.1234/test.001",
        ),
        MockDocumentMetadata(
            doc_id="doc_002",
            title="Deep Learning Advances",
            authors='["Martin, P."]',
            year=2023,
            journal="Nature AI",
            volume="15",
            doi="10.1234/test.002",
        ),
        MockDocumentMetadata(
            doc_id="doc_003",
            title="Rapport annuel 2024",
            authors=None,
            year=2024,
            publisher="Ministère de l'Éducation",
            doc_type="report",
        ),
    ]

    def get_doc(doc_id):
        for d in store.get_all_documents():
            if d.doc_id == doc_id:
                return d
        return None

    store.get_document.side_effect = get_doc
    return store


@pytest.fixture
def engine(mock_store):
    return CitationEngine(metadata_store=mock_store, enabled=True)


class TestCitationPipeline:
    def test_full_pipeline_extract_resolve_compile(self, engine):
        """Pipeline complet : extraction → résolution → bibliographie."""
        text = (
            "Selon Dupont (Dupont, 2024), le machine learning révolutionne "
            "l'éducation. Martin (Martin, 2023) confirme ces avancées."
        )

        # Step 1: Extract
        citations = CitationEngine.extract_inline_citations(text)
        assert len(citations) == 2

        # Step 2: Resolve
        resolved = engine.resolve_citations(citations)
        dupont_cit = next(c for c in resolved if "Dupont" in c.authors)
        martin_cit = next(c for c in resolved if "Martin" in c.authors)
        assert dupont_cit.resolved_doc_id == "doc_001"
        assert martin_cit.resolved_doc_id == "doc_002"

        # Step 3: Compile bibliography
        bibliography = engine.compile_bibliography()
        assert len(bibliography) == 2
        assert all(isinstance(e, BibliographyEntry) for e in bibliography)

        # Check alphabetical order
        refs = [e.apa_reference for e in bibliography]
        assert refs == sorted(refs, key=str.lower)

    def test_pipeline_unresolved_citation(self, engine):
        """Citation non résolue n'apparaît pas dans la bibliographie."""
        text = "Selon (Inconnu, 2099), les résultats sont surprenants."
        citations = CitationEngine.extract_inline_citations(text)
        assert len(citations) == 1

        resolved = engine.resolve_citations(citations)
        assert resolved[0].resolved_doc_id is None

        bibliography = engine.compile_bibliography()
        assert len(bibliography) == 0

    def test_pipeline_multiple_sections(self, engine):
        """Les citations s'accumulent entre les sections."""
        text1 = "Recherche (Dupont, 2024) en éducation."
        text2 = "Avancées (Martin, 2023) en deep learning."

        cit1 = CitationEngine.extract_inline_citations(text1)
        engine.resolve_citations(cit1)

        cit2 = CitationEngine.extract_inline_citations(text2)
        engine.resolve_citations(cit2)

        bibliography = engine.compile_bibliography()
        assert len(bibliography) == 2

    def test_pipeline_reset(self, engine):
        """Reset efface les citations accumulées."""
        text = "Citation (Dupont, 2024) et (Martin, 2023)."
        citations = CitationEngine.extract_inline_citations(text)
        engine.resolve_citations(citations)
        assert len(engine.get_cited_doc_ids()) == 2

        engine.reset()
        assert len(engine.get_cited_doc_ids()) == 0

    def test_apa_article_format(self):
        """Format APA pour un article de revue."""
        ref = CitationEngine.format_apa_reference(
            doc_type="article",
            authors="Dupont, J., Smith, A.",
            year=2024,
            title="Machine Learning for Education",
            journal="Journal of AI",
            volume="42",
            issue="3",
            pages_range="100-120",
            doi="10.1234/test.001",
        )
        assert "Dupont, J., Smith, A." in ref
        assert "(2024)" in ref
        assert "*Journal of AI*" in ref
        assert "*42*" in ref
        assert "(3)" in ref
        assert "100-120" in ref
        assert "https://doi.org/10.1234/test.001" in ref

    def test_apa_book_format(self):
        """Format APA pour un livre."""
        ref = CitationEngine.format_apa_reference(
            doc_type="book",
            authors="Author, A.",
            year=2020,
            title="Big Book Title",
            publisher="Publisher Inc.",
        )
        assert "*Big Book Title*" in ref
        assert "Publisher Inc." in ref

    def test_bibliography_marks_incomplete(self, engine, mock_store):
        """Les métadonnées incomplètes sont marquées."""
        engine._cited_doc_ids.add("doc_003")
        bibliography = engine.compile_bibliography()
        incomplete = [e for e in bibliography if "incomplètes" in e.apa_reference]
        assert len(incomplete) == 1

    def test_disabled_engine(self, mock_store):
        """Moteur désactivé ne résout pas et ne compile pas."""
        engine = CitationEngine(metadata_store=mock_store, enabled=False)
        citations = [CitationRef(raw_text="(Test, 2024)", authors="Test", year=2024)]
        resolved = engine.resolve_citations(citations)
        assert all(c.resolved_doc_id is None for c in resolved)
        assert engine.compile_bibliography() == []
