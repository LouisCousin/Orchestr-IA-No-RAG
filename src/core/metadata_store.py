"""Base SQLite de métadonnées riches pour le corpus.

Phase 2.5 : stocke les métadonnées complètes de chaque document et chunk
dans une base SQLite locale, complémentaire à ChromaDB (qui ne stocke
que des métadonnées simples).

Prépare les fondations pour les citations APA (Phase 3).
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestria")


@dataclass
class DocumentMetadata:
    """Métadonnées complètes d'un document du corpus."""
    doc_id: str
    filepath: str
    filename: str
    title: Optional[str] = None
    authors: Optional[str] = None  # JSON array : '["Dupont, J.", "Smith, A."]'
    year: Optional[int] = None
    language: str = "fr"
    doc_type: str = "unknown"  # "article", "report", "thesis", "book", "web"
    page_count: int = 0
    token_count: int = 0
    char_count: int = 0
    word_count: int = 0
    extraction_method: Optional[str] = None
    extraction_status: Optional[str] = None
    hash_binary: Optional[str] = None
    hash_textual: Optional[str] = None
    dedup_status: str = "unique"
    dedup_original_id: Optional[str] = None
    # Champs Phase 3 (préremplis vides)
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages_range: Optional[str] = None
    doi: Optional[str] = None
    publisher: Optional[str] = None
    apa_reference: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filepath TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    authors TEXT,
    year INTEGER,
    language TEXT DEFAULT 'fr',
    doc_type TEXT DEFAULT 'unknown',
    page_count INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    extraction_method TEXT,
    extraction_status TEXT,
    hash_binary TEXT,
    hash_textual TEXT,
    dedup_status TEXT DEFAULT 'unique',
    dedup_original_id TEXT,
    journal TEXT,
    volume TEXT,
    issue TEXT,
    pages_range TEXT,
    doi TEXT,
    publisher TEXT,
    apa_reference TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    chunk_index INTEGER,
    token_count INTEGER,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_language ON documents(language);
CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(hash_textual);
"""


class MetadataStore:
    """Interface SQLite pour les métadonnées riches du corpus."""

    def __init__(self, project_path: str):
        self.db_path = os.path.join(project_path, "metadata.db")
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        """Initialise le schéma de la base."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        """Ferme la connexion."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Documents CRUD ──

    def add_document(self, doc: DocumentMetadata) -> None:
        """Ajoute un document à la base."""
        conn = self._get_conn()
        now = datetime.now().isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO documents
            (doc_id, filepath, filename, title, authors, year, language,
             doc_type, page_count, token_count, char_count, word_count,
             extraction_method, extraction_status, hash_binary, hash_textual,
             dedup_status, dedup_original_id, journal, volume, issue,
             pages_range, doi, publisher, apa_reference, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.doc_id, doc.filepath, doc.filename, doc.title, doc.authors,
                doc.year, doc.language, doc.doc_type, doc.page_count,
                doc.token_count, doc.char_count, doc.word_count,
                doc.extraction_method, doc.extraction_status,
                doc.hash_binary, doc.hash_textual, doc.dedup_status,
                doc.dedup_original_id, doc.journal, doc.volume, doc.issue,
                doc.pages_range, doc.doi, doc.publisher, doc.apa_reference,
                doc.created_at or now, now,
            ),
        )
        conn.commit()

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Récupère un document par son ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def get_all_documents(self) -> list[DocumentMetadata]:
        """Récupère tous les documents."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM documents ORDER BY filename").fetchall()
        return [self._row_to_document(r) for r in rows]

    def update_document(self, doc_id: str, **fields) -> None:
        """Met à jour les champs spécifiés d'un document."""
        if not fields:
            return
        conn = self._get_conn()
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [doc_id]
        conn.execute(f"UPDATE documents SET {set_clause} WHERE doc_id = ?", values)
        conn.commit()

    def delete_document(self, doc_id: str) -> None:
        """Supprime un document et ses chunks."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()

    def search_documents(
        self,
        language: Optional[str] = None,
        doc_type: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> list[DocumentMetadata]:
        """Recherche filtrée de documents."""
        conn = self._get_conn()
        conditions = []
        params = []

        if language:
            conditions.append("language = ?")
            params.append(language)
        if doc_type:
            conditions.append("doc_type = ?")
            params.append(doc_type)
        if year_min is not None:
            conditions.append("year >= ?")
            params.append(year_min)
        if year_max is not None:
            conditions.append("year <= ?")
            params.append(year_max)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(f"SELECT * FROM documents WHERE {where}", params).fetchall()
        return [self._row_to_document(r) for r in rows]

    def get_doc_ids_by_filter(self, **filters) -> list[str]:
        """Retourne les doc_id matchant les filtres (pour pré-filtrage ChromaDB)."""
        conn = self._get_conn()
        conditions = []
        params = []

        for key, value in filters.items():
            if value is not None:
                conditions.append(f"{key} = ?")
                params.append(value)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(f"SELECT doc_id FROM documents WHERE {where}", params).fetchall()
        return [r["doc_id"] for r in rows]

    # ── Chunks CRUD ──

    def add_chunks(self, chunks: list) -> None:
        """Ajoute une liste de chunks à la base.

        Args:
            chunks: Liste d'objets Chunk (du module semantic_chunker).
        """
        conn = self._get_conn()
        for chunk in chunks:
            conn.execute(
                """INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, text, page_number, section_title, chunk_index, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk.chunk_id, chunk.doc_id, chunk.text,
                    chunk.page_number, chunk.section_title,
                    chunk.chunk_index, chunk.token_count,
                ),
            )
        conn.commit()

    def get_chunks_by_doc(self, doc_id: str) -> list[dict]:
        """Récupère tous les chunks d'un document, triés par index."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Récupère un chunk par son ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def get_all_chunks(self) -> list[dict]:
        """Récupère tous les chunks."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_index").fetchall()
        return [dict(r) for r in rows]

    def count_chunks(self) -> int:
        """Nombre total de chunks."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"]

    # ── Phase 3 methods ──

    def update_from_grobid(self, doc_id: str, grobid_data: dict) -> None:
        """Met à jour les champs bibliographiques depuis GROBID."""
        fields = {}
        if grobid_data.get("title"):
            fields["title"] = grobid_data["title"]
        if grobid_data.get("authors"):
            fields["authors"] = json.dumps(grobid_data["authors"])
        if grobid_data.get("year"):
            fields["year"] = grobid_data["year"]
        if grobid_data.get("journal"):
            fields["journal"] = grobid_data["journal"]
        if grobid_data.get("volume"):
            fields["volume"] = grobid_data["volume"]
        if grobid_data.get("issue"):
            fields["issue"] = grobid_data["issue"]
        if grobid_data.get("pages"):
            fields["pages_range"] = grobid_data["pages"]
        if grobid_data.get("doi"):
            fields["doi"] = grobid_data["doi"]
        if grobid_data.get("publisher"):
            fields["publisher"] = grobid_data["publisher"]
        if fields:
            self.update_document(doc_id, **fields)

    def apply_overrides(self, doc_id: str, override_data: dict) -> None:
        """Applique les corrections YAML d'overrides de métadonnées."""
        fields = {}
        for key, value in override_data.items():
            if key == "authors" and isinstance(value, list):
                fields["authors"] = json.dumps(value)
            elif value is not None:
                fields[key] = value
        if fields:
            self.update_document(doc_id, **fields)

    def get_apa_reference(self, doc_id: str) -> Optional[str]:
        """Retourne la référence APA formatée d'un document."""
        doc = self.get_document(doc_id)
        return doc.apa_reference if doc else None

    def get_cited_documents(self, doc_ids: list) -> list[DocumentMetadata]:
        """Récupère les métadonnées des documents cités."""
        return [doc for doc_id in doc_ids if (doc := self.get_document(doc_id)) is not None]

    # ── Helpers ──

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentMetadata:
        return DocumentMetadata(
            doc_id=row["doc_id"],
            filepath=row["filepath"],
            filename=row["filename"],
            title=row["title"],
            authors=row["authors"],
            year=row["year"],
            language=row["language"],
            doc_type=row["doc_type"],
            page_count=row["page_count"],
            token_count=row["token_count"],
            char_count=row["char_count"],
            word_count=row["word_count"],
            extraction_method=row["extraction_method"],
            extraction_status=row["extraction_status"],
            hash_binary=row["hash_binary"],
            hash_textual=row["hash_textual"],
            dedup_status=row["dedup_status"],
            dedup_original_id=row["dedup_original_id"],
            journal=row["journal"],
            volume=row["volume"],
            issue=row["issue"],
            pages_range=row["pages_range"],
            doi=row["doi"],
            publisher=row["publisher"],
            apa_reference=row["apa_reference"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
