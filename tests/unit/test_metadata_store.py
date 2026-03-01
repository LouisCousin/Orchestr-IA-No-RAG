"""Tests unitaires pour le module metadata_store (Phase 2.5)."""

import pytest
import os
from dataclasses import dataclass

from src.core.metadata_store import MetadataStore, DocumentMetadata


@dataclass
class Chunk:
    """Mock de Chunk (v4.0 : semantic_chunker supprimé)."""
    doc_id: str = ""
    text: str = ""
    page_number: int = 0
    section_title: str = ""
    chunk_index: int = 0
    token_count: int = 0
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}_{self.chunk_index}"


@pytest.fixture
def store(tmp_path):
    """Crée un MetadataStore avec une base en mémoire temporaire."""
    s = MetadataStore(str(tmp_path))
    yield s
    s.close()


@pytest.fixture
def sample_doc():
    return DocumentMetadata(
        doc_id="doc001",
        filepath="/corpus/rapport.pdf",
        filename="rapport.pdf",
        title="Rapport annuel 2024",
        authors='["Dupont, J.", "Martin, A."]',
        year=2024,
        language="fr",
        doc_type="report",
        page_count=42,
        token_count=15000,
        char_count=60000,
        word_count=10000,
        extraction_method="docling",
        extraction_status="success",
        hash_binary="abc123",
        hash_textual="def456",
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(doc_id="doc001", text="Contenu du premier chunk.", page_number=1,
              section_title="Introduction", chunk_index=0),
        Chunk(doc_id="doc001", text="Contenu du deuxième chunk.", page_number=2,
              section_title="Méthodologie", chunk_index=1),
        Chunk(doc_id="doc001", text="Contenu du troisième chunk.", page_number=3,
              section_title="Résultats", chunk_index=2),
    ]


# ── Tests DB initialisation ──

class TestInit:
    def test_db_file_created(self, store, tmp_path):
        assert os.path.exists(os.path.join(str(tmp_path), "metadata.db"))

    def test_tables_exist(self, store):
        conn = store._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "documents" in table_names
        assert "chunks" in table_names


# ── Tests Documents CRUD ──

class TestDocumentCRUD:
    def test_add_and_get_document(self, store, sample_doc):
        store.add_document(sample_doc)
        retrieved = store.get_document("doc001")
        assert retrieved is not None
        assert retrieved.doc_id == "doc001"
        assert retrieved.title == "Rapport annuel 2024"
        assert retrieved.year == 2024
        assert retrieved.language == "fr"
        assert retrieved.doc_type == "report"
        assert retrieved.page_count == 42
        assert retrieved.token_count == 15000
        assert retrieved.extraction_method == "docling"

    def test_get_nonexistent_document(self, store):
        result = store.get_document("nonexistent")
        assert result is None

    def test_get_all_documents(self, store, sample_doc):
        store.add_document(sample_doc)
        doc2 = DocumentMetadata(
            doc_id="doc002", filepath="/corpus/etude.pdf",
            filename="etude.pdf", title="Étude de marché",
            language="fr", doc_type="article",
        )
        store.add_document(doc2)
        docs = store.get_all_documents()
        assert len(docs) == 2

    def test_update_document(self, store, sample_doc):
        store.add_document(sample_doc)
        store.update_document("doc001", title="Titre modifié", year=2025)
        updated = store.get_document("doc001")
        assert updated.title == "Titre modifié"
        assert updated.year == 2025
        # Les autres champs restent inchangés
        assert updated.language == "fr"

    def test_update_empty_fields(self, store, sample_doc):
        store.add_document(sample_doc)
        store.update_document("doc001")  # No fields
        unchanged = store.get_document("doc001")
        assert unchanged.title == "Rapport annuel 2024"

    def test_delete_document(self, store, sample_doc, sample_chunks):
        store.add_document(sample_doc)
        store.add_chunks(sample_chunks)
        assert store.get_document("doc001") is not None
        assert store.count_chunks() == 3

        store.delete_document("doc001")
        assert store.get_document("doc001") is None
        assert store.count_chunks() == 0

    def test_add_document_replace(self, store, sample_doc):
        store.add_document(sample_doc)
        sample_doc.title = "Nouveau titre"
        store.add_document(sample_doc)
        retrieved = store.get_document("doc001")
        assert retrieved.title == "Nouveau titre"
        docs = store.get_all_documents()
        assert len(docs) == 1


# ── Tests search_documents ──

class TestSearchDocuments:
    def _populate(self, store):
        docs = [
            DocumentMetadata(doc_id="d1", filepath="a.pdf", filename="a.pdf",
                             language="fr", doc_type="article", year=2022),
            DocumentMetadata(doc_id="d2", filepath="b.pdf", filename="b.pdf",
                             language="en", doc_type="report", year=2023),
            DocumentMetadata(doc_id="d3", filepath="c.pdf", filename="c.pdf",
                             language="fr", doc_type="report", year=2024),
            DocumentMetadata(doc_id="d4", filepath="d.pdf", filename="d.pdf",
                             language="de", doc_type="thesis", year=2021),
        ]
        for d in docs:
            store.add_document(d)

    def test_search_by_language(self, store):
        self._populate(store)
        results = store.search_documents(language="fr")
        assert len(results) == 2
        assert all(r.language == "fr" for r in results)

    def test_search_by_doc_type(self, store):
        self._populate(store)
        results = store.search_documents(doc_type="report")
        assert len(results) == 2
        assert all(r.doc_type == "report" for r in results)

    def test_search_by_year_range(self, store):
        self._populate(store)
        results = store.search_documents(year_min=2023, year_max=2024)
        assert len(results) == 2
        assert all(2023 <= r.year <= 2024 for r in results)

    def test_search_combined_filters(self, store):
        self._populate(store)
        results = store.search_documents(language="fr", doc_type="report")
        assert len(results) == 1
        assert results[0].doc_id == "d3"

    def test_search_no_filters(self, store):
        self._populate(store)
        results = store.search_documents()
        assert len(results) == 4

    def test_search_no_results(self, store):
        self._populate(store)
        results = store.search_documents(language="jp")
        assert len(results) == 0


class TestGetDocIdsByFilter:
    def test_filter_by_language(self, store):
        store.add_document(DocumentMetadata(
            doc_id="d1", filepath="a.pdf", filename="a.pdf", language="fr",
        ))
        store.add_document(DocumentMetadata(
            doc_id="d2", filepath="b.pdf", filename="b.pdf", language="en",
        ))
        ids = store.get_doc_ids_by_filter(language="fr")
        assert ids == ["d1"]

    def test_filter_no_match(self, store):
        store.add_document(DocumentMetadata(
            doc_id="d1", filepath="a.pdf", filename="a.pdf", language="fr",
        ))
        ids = store.get_doc_ids_by_filter(language="de")
        assert ids == []


# ── Tests Chunks CRUD ──

class TestChunksCRUD:
    def test_add_and_get_chunks(self, store, sample_doc, sample_chunks):
        store.add_document(sample_doc)
        store.add_chunks(sample_chunks)
        chunks = store.get_chunks_by_doc("doc001")
        assert len(chunks) == 3
        assert chunks[0]["section_title"] == "Introduction"
        assert chunks[1]["section_title"] == "Méthodologie"
        assert chunks[2]["section_title"] == "Résultats"

    def test_get_chunk_by_id(self, store, sample_doc, sample_chunks):
        store.add_document(sample_doc)
        store.add_chunks(sample_chunks)
        chunk = store.get_chunk(sample_chunks[0].chunk_id)
        assert chunk is not None
        assert chunk["doc_id"] == "doc001"
        assert chunk["text"] == "Contenu du premier chunk."

    def test_get_nonexistent_chunk(self, store):
        result = store.get_chunk("nonexistent")
        assert result is None

    def test_get_all_chunks(self, store, sample_doc, sample_chunks):
        store.add_document(sample_doc)
        store.add_chunks(sample_chunks)
        all_chunks = store.get_all_chunks()
        assert len(all_chunks) == 3

    def test_count_chunks(self, store, sample_doc, sample_chunks):
        store.add_document(sample_doc)
        store.add_chunks(sample_chunks)
        assert store.count_chunks() == 3

    def test_count_chunks_empty(self, store):
        assert store.count_chunks() == 0

    def test_chunks_ordered_by_index(self, store, sample_doc, sample_chunks):
        store.add_document(sample_doc)
        store.add_chunks(sample_chunks)
        chunks = store.get_chunks_by_doc("doc001")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == [0, 1, 2]

    def test_delete_document_cascades_chunks(self, store, sample_doc, sample_chunks):
        store.add_document(sample_doc)
        store.add_chunks(sample_chunks)
        store.delete_document("doc001")
        chunks = store.get_chunks_by_doc("doc001")
        assert len(chunks) == 0


# ── Tests Phase 3 fields ──

class TestPhase3Fields:
    def test_phase3_fields_stored(self, store):
        doc = DocumentMetadata(
            doc_id="d1", filepath="a.pdf", filename="a.pdf",
            journal="Nature", volume="612", issue="3",
            pages_range="42-56", doi="10.1038/s41586-024-00001-z",
            publisher="Springer", apa_reference="Dupont, J. (2024). ...",
        )
        store.add_document(doc)
        retrieved = store.get_document("d1")
        assert retrieved.journal == "Nature"
        assert retrieved.volume == "612"
        assert retrieved.doi == "10.1038/s41586-024-00001-z"
        assert retrieved.apa_reference == "Dupont, J. (2024). ..."


# ── Test close ──

class TestClose:
    def test_close_and_reopen(self, tmp_path, sample_doc):
        store = MetadataStore(str(tmp_path))
        store.add_document(sample_doc)
        store.close()

        store2 = MetadataStore(str(tmp_path))
        retrieved = store2.get_document("doc001")
        assert retrieved is not None
        assert retrieved.title == "Rapport annuel 2024"
        store2.close()
