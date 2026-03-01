"""Tests unitaires pour atlas_builder.py — v4.1."""

import json
import pytest

from src.core.atlas_builder import (
    AtlasFact,
    AtlasDocumentEntry,
    AtlasCrossIndex,
    CorpusAtlas,
    AtlasBuilder,
)


class TestAtlasFact:
    def test_basic_creation(self):
        fact = AtlasFact(text="Le PIB est de 3%", source_file="doc.pdf")
        assert fact.text == "Le PIB est de 3%"
        assert fact.source_file == "doc.pdf"
        assert fact.confidence == 1.0
        assert fact.page is None


class TestAtlasDocumentEntry:
    def test_to_dict_roundtrip(self):
        entry = AtlasDocumentEntry(
            source_file="rapport.pdf",
            doc_index=0,
            title="Rapport annuel",
            summary="Résumé du rapport",
            themes=["finance", "croissance"],
            entities=["Société X", "Paris"],
            key_facts=[
                AtlasFact(text="CA +5%", source_file="rapport.pdf", page=3),
            ],
            token_count=10000,
            factual_density=1.5,
        )
        d = entry.to_dict()
        assert d["source_file"] == "rapport.pdf"
        assert d["title"] == "Rapport annuel"
        assert len(d["key_facts"]) == 1
        assert d["key_facts"][0]["page"] == 3

        # Roundtrip
        restored = AtlasDocumentEntry.from_dict(d)
        assert restored.source_file == "rapport.pdf"
        assert restored.title == "Rapport annuel"
        assert len(restored.key_facts) == 1
        assert restored.key_facts[0].page == 3
        assert restored.factual_density == 1.5

    def test_from_dict_defaults(self):
        entry = AtlasDocumentEntry.from_dict({"source_file": "test.pdf"})
        assert entry.title == ""
        assert entry.themes == []
        assert entry.key_facts == []
        assert entry.token_count == 0


class TestAtlasCrossIndex:
    def test_to_dict_roundtrip(self):
        cross = AtlasCrossIndex(
            entity_to_docs={"paris": ["doc1.pdf", "doc2.pdf"]},
            theme_to_docs={"finance": ["doc1.pdf"]},
            contradictions=[{
                "entity": "taux",
                "doc_a": "doc1.pdf",
                "doc_b": "doc2.pdf",
                "description": "Divergence sur le taux",
            }],
        )
        d = cross.to_dict()
        restored = AtlasCrossIndex.from_dict(d)
        assert restored.entity_to_docs == {"paris": ["doc1.pdf", "doc2.pdf"]}
        assert len(restored.contradictions) == 1


class TestCorpusAtlas:
    def _make_atlas(self):
        return CorpusAtlas(
            documents=[
                AtlasDocumentEntry(
                    source_file="doc1.pdf",
                    doc_index=0,
                    title="Premier doc",
                    summary="Résumé 1",
                    themes=["tech", "IA"],
                    entities=["OpenAI", "Google"],
                    key_facts=[
                        AtlasFact(text="GPT-4 lancé en 2023", source_file="doc1.pdf"),
                    ],
                    token_count=5000,
                    factual_density=2.0,
                ),
                AtlasDocumentEntry(
                    source_file="doc2.pdf",
                    doc_index=1,
                    title="Second doc",
                    summary="Résumé 2",
                    themes=["finance", "tech"],
                    entities=["Google", "Apple"],
                    key_facts=[
                        AtlasFact(text="CA $100M", source_file="doc2.pdf"),
                    ],
                    token_count=8000,
                    factual_density=1.0,
                ),
            ],
            cross_index=AtlasCrossIndex(
                entity_to_docs={"google": ["doc1.pdf", "doc2.pdf"]},
                theme_to_docs={"tech": ["doc1.pdf", "doc2.pdf"]},
            ),
            total_corpus_tokens=13000,
            atlas_tokens=500,
            compression_ratio=26.0,
            indexation_model="gemini-3-flash-preview",
        )

    def test_to_dict_roundtrip(self):
        atlas = self._make_atlas()
        d = atlas.to_dict()
        restored = CorpusAtlas.from_dict(d)

        assert len(restored.documents) == 2
        assert restored.documents[0].source_file == "doc1.pdf"
        assert restored.total_corpus_tokens == 13000
        assert restored.compression_ratio == 26.0
        assert "google" in restored.cross_index.entity_to_docs

    def test_get_doc_by_source(self):
        atlas = self._make_atlas()
        doc = atlas.get_doc_by_source("doc1.pdf")
        assert doc is not None
        assert doc.title == "Premier doc"

        missing = atlas.get_doc_by_source("nonexistent.pdf")
        assert missing is None

    def test_format_for_prompt(self):
        atlas = self._make_atlas()
        text = atlas.format_for_prompt()
        assert "CORPUS ATLAS" in text
        assert "doc1.pdf" in text
        assert "Résumé 1" in text
        assert "tech" in text

    def test_format_for_prompt_with_contradictions(self):
        atlas = self._make_atlas()
        atlas.cross_index.contradictions = [{
            "entity": "taux",
            "description": "Divergence détectée",
        }]
        text = atlas.format_for_prompt()
        assert "CONTRADICTIONS" in text
        assert "Divergence" in text


class TestAtlasBuilderTopK:
    def _make_atlas(self):
        return CorpusAtlas(
            documents=[
                AtlasDocumentEntry(
                    source_file="finance.pdf",
                    doc_index=0,
                    title="Rapport financier",
                    summary="Analyse financière",
                    themes=["finance", "investissement"],
                    entities=["BNP", "Société Générale"],
                    key_facts=[
                        AtlasFact(text="ROE 12%", source_file="finance.pdf"),
                        AtlasFact(text="CA +8%", source_file="finance.pdf"),
                    ],
                    token_count=10000,
                    factual_density=3.0,
                ),
                AtlasDocumentEntry(
                    source_file="tech.pdf",
                    doc_index=1,
                    title="Rapport tech",
                    summary="Innovation IA",
                    themes=["technologie", "IA"],
                    entities=["OpenAI", "Google"],
                    key_facts=[
                        AtlasFact(text="GPT-5 annoncé", source_file="tech.pdf"),
                    ],
                    token_count=8000,
                    factual_density=1.5,
                ),
                AtlasDocumentEntry(
                    source_file="rh.pdf",
                    doc_index=2,
                    title="Rapport RH",
                    summary="Gestion des talents",
                    themes=["ressources humaines", "recrutement"],
                    entities=["LinkedIn"],
                    key_facts=[],
                    token_count=5000,
                    factual_density=0.0,
                ),
            ],
            total_corpus_tokens=23000,
        )

    def test_top_k_thematic_match(self):
        atlas = self._make_atlas()
        result = AtlasBuilder.select_top_k_documents(
            atlas=atlas,
            section_title="Analyse financière et investissement",
            section_themes=["finance", "investissement"],
            section_entities=[],
            k=2,
        )
        assert result[0] == "finance.pdf"
        assert len(result) == 2

    def test_top_k_entity_match(self):
        atlas = self._make_atlas()
        result = AtlasBuilder.select_top_k_documents(
            atlas=atlas,
            section_title="Stratégie Google",
            section_themes=[],
            section_entities=["Google"],
            k=1,
        )
        assert result[0] == "tech.pdf"

    def test_top_k_empty_atlas(self):
        atlas = CorpusAtlas()
        result = AtlasBuilder.select_top_k_documents(
            atlas=atlas,
            section_title="Test",
            section_themes=["test"],
            section_entities=[],
            k=3,
        )
        assert result == []

    def test_top_k_contradiction_bonus(self):
        atlas = self._make_atlas()
        atlas.cross_index.contradictions = [{
            "entity": "taux",
            "doc_a": "rh.pdf",
            "doc_b": "tech.pdf",
            "description": "Divergence",
        }]
        result = AtlasBuilder.select_top_k_documents(
            atlas=atlas,
            section_title="Innovation",
            section_themes=["technologie"],
            section_entities=[],
            k=3,
        )
        # tech.pdf should score higher thanks to thematic + contradiction bonus
        assert "tech.pdf" in result[:2]

    def test_top_k_respects_k_limit(self):
        atlas = self._make_atlas()
        result = AtlasBuilder.select_top_k_documents(
            atlas=atlas,
            section_title="Tout",
            section_themes=["finance", "technologie", "rh"],
            section_entities=[],
            k=1,
        )
        assert len(result) == 1


class TestAtlasBuilderParsing:
    def test_parse_indexation_response_valid_json(self):
        builder = AtlasBuilder(provider=None, model="test")
        content = json.dumps({
            "title": "Mon Document",
            "summary": "Un résumé",
            "themes": ["tech"],
            "entities": ["Google"],
            "key_facts": [{"text": "Fait 1", "page": 5}],
        })
        entry = builder._parse_indexation_response(
            content, "doc.pdf", 0, 5000
        )
        assert entry.title == "Mon Document"
        assert entry.summary == "Un résumé"
        assert entry.themes == ["tech"]
        assert len(entry.key_facts) == 1
        assert entry.key_facts[0].page == 5

    def test_parse_indexation_response_markdown_wrapped(self):
        builder = AtlasBuilder(provider=None, model="test")
        content = '```json\n{"title": "Test", "summary": "S", "themes": [], "entities": [], "key_facts": []}\n```'
        entry = builder._parse_indexation_response(
            content, "doc.pdf", 0, 1000
        )
        assert entry.title == "Test"

    def test_parse_indexation_response_invalid_json(self):
        builder = AtlasBuilder(provider=None, model="test")
        entry = builder._parse_indexation_response(
            "not json at all!", "doc.pdf", 0, 1000
        )
        assert entry.source_file == "doc.pdf"
        assert "Parsing échoué" in entry.summary

    def test_parse_indexation_response_string_facts(self):
        builder = AtlasBuilder(provider=None, model="test")
        content = json.dumps({
            "title": "Doc",
            "summary": "S",
            "themes": [],
            "entities": [],
            "key_facts": ["fait simple en string"],
        })
        entry = builder._parse_indexation_response(
            content, "doc.pdf", 0, 1000
        )
        assert len(entry.key_facts) == 1
        assert entry.key_facts[0].text == "fait simple en string"


class TestAtlasBuilderCrossIndex:
    def test_build_cross_index(self):
        builder = AtlasBuilder(provider=None, model="test")
        docs = [
            AtlasDocumentEntry(
                source_file="doc1.pdf", doc_index=0,
                themes=["finance", "tech"],
                entities=["Google", "BNP"],
                key_facts=[],
            ),
            AtlasDocumentEntry(
                source_file="doc2.pdf", doc_index=1,
                themes=["tech", "IA"],
                entities=["Google", "OpenAI"],
                key_facts=[],
            ),
        ]
        cross = builder._build_cross_index(docs)
        assert "google" in cross.entity_to_docs
        assert len(cross.entity_to_docs["google"]) == 2
        assert "tech" in cross.theme_to_docs
        assert len(cross.theme_to_docs["tech"]) == 2

    def test_build_cross_index_no_duplicates(self):
        builder = AtlasBuilder(provider=None, model="test")
        docs = [
            AtlasDocumentEntry(
                source_file="doc1.pdf", doc_index=0,
                entities=["X", "X"],  # duplicate entity in same doc
                themes=[],
                key_facts=[],
            ),
        ]
        cross = builder._build_cross_index(docs)
        assert len(cross.entity_to_docs.get("x", [])) == 1
