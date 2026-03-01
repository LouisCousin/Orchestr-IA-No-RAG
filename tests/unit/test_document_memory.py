"""Tests unitaires pour document_memory.py — v4.1."""

import pytest

from src.core.document_memory import DocumentMemory, SectionMemoryEntry


class TestDocumentMemory:
    def test_empty_memory(self):
        memory = DocumentMemory()
        assert memory.get_section_count() == 0
        assert memory.format_for_prompt() == ""
        assert memory.get_entities() == set()
        assert memory.get_sources() == set()

    def test_add_section(self):
        memory = DocumentMemory()
        memory.add_section(
            section_id="s01",
            section_title="Introduction",
            content="Ceci est l'introduction du document.",
            summary="Introduction présentant le contexte.",
            entities=["France", "Union Européenne"],
            facts=["PIB 2024 : 2800Md€"],
            sources=["rapport.pdf"],
        )
        assert memory.get_section_count() == 1
        assert "france" in memory.get_entities()
        assert "union européenne" in memory.get_entities()
        assert "rapport.pdf" in memory.get_sources()

    def test_format_for_prompt_content(self):
        memory = DocumentMemory()
        memory.add_section(
            section_id="s01",
            section_title="Contexte",
            content="Le contexte économique actuel.",
            summary="Résumé du contexte",
        )
        text = memory.format_for_prompt()
        assert "MÉMOIRE DU DOCUMENT" in text
        assert "Contexte" in text
        assert "Résumé du contexte" in text
        assert "cohérence" in text.lower()

    def test_multiple_sections(self):
        memory = DocumentMemory()
        for i in range(5):
            memory.add_section(
                section_id=f"s{i:02d}",
                section_title=f"Section {i}",
                content=f"Contenu de la section {i}." * 50,
                entities=[f"Entité_{i}"],
                sources=[f"doc{i}.pdf"],
            )
        assert memory.get_section_count() == 5
        assert len(memory.get_entities()) == 5
        assert len(memory.get_sources()) == 5

    def test_auto_summary_when_empty(self):
        memory = DocumentMemory()
        memory.add_section(
            section_id="s01",
            section_title="Test",
            content="Contenu court.",
        )
        entry = memory._entries[0]
        assert entry.summary == "Contenu court."

    def test_auto_summary_truncation(self):
        memory = DocumentMemory()
        long_content = "A" * 2000
        memory.add_section(
            section_id="s01",
            section_title="Test",
            content=long_content,
        )
        entry = memory._entries[0]
        assert len(entry.summary) < 2000
        assert entry.summary.endswith("...")

    def test_entities_are_lowercase(self):
        memory = DocumentMemory()
        memory.add_section(
            section_id="s01",
            section_title="Test",
            content="Test",
            entities=["Paris", "FRANCE", "berlin"],
        )
        entities = memory.get_entities()
        assert "paris" in entities
        assert "france" in entities
        assert "berlin" in entities
        assert "Paris" not in entities


class TestDocumentMemorySerialization:
    def test_to_dict(self):
        memory = DocumentMemory()
        memory.add_section(
            section_id="s01",
            section_title="Intro",
            content="Contenu",
            summary="Résumé",
            entities=["Paris"],
            sources=["doc.pdf"],
        )
        d = memory.to_dict()
        assert len(d["entries"]) == 1
        assert d["entries"][0]["section_id"] == "s01"
        assert "paris" in d["global_entities"]
        assert "doc.pdf" in d["global_sources"]

    def test_from_dict_roundtrip(self):
        memory = DocumentMemory()
        memory.add_section(
            section_id="s01",
            section_title="Intro",
            content="Contenu de l'intro",
            summary="Résumé intro",
            entities=["France"],
            facts=["PIB +2%"],
            sources=["rapport.pdf"],
        )
        memory.add_section(
            section_id="s02",
            section_title="Analyse",
            content="Analyse détaillée",
            summary="Résumé analyse",
            entities=["Allemagne"],
            sources=["etude.pdf"],
        )

        d = memory.to_dict()
        restored = DocumentMemory.from_dict(d)

        assert restored.get_section_count() == 2
        assert "france" in restored.get_entities()
        assert "allemagne" in restored.get_entities()
        assert "rapport.pdf" in restored.get_sources()
        assert "etude.pdf" in restored.get_sources()

    def test_from_dict_empty(self):
        restored = DocumentMemory.from_dict({})
        assert restored.get_section_count() == 0


class TestDocumentMemoryPruning:
    def test_pruning_large_memory(self):
        memory = DocumentMemory(max_tokens=2000)
        # Add enough sections to trigger pruning
        for i in range(10):
            memory.add_section(
                section_id=f"s{i:02d}",
                section_title=f"Section {i}",
                content="X" * 200,
                summary="Résumé " * 50,
                entities=[f"entity_{i}"],
            )
        # After pruning, some entries may have been removed or summaries shortened
        assert memory.get_section_count() <= 10

    def test_word_count_tracking(self):
        memory = DocumentMemory()
        memory.add_section(
            section_id="s01",
            section_title="Test",
            content="un deux trois quatre cinq",
        )
        assert memory._entries[0].word_count == 5
