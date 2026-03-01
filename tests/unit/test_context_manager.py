"""Tests unitaires pour le ContextManager — v4.0."""

import pytest

from src.core.context_manager import AgentContextPayload, ContextManager
from src.core.strategy_selector import GenerationStrategy


class TestAgentContextPayload:
    """Tests du payload de contexte agent."""

    def test_default_values(self):
        payload = AgentContextPayload()
        assert payload.corpus_text is None
        assert payload.cache_name is None
        assert payload.strategy == "standard"

    def test_standard_payload(self):
        payload = AgentContextPayload(
            corpus_text="<corpus>test</corpus>",
            strategy="standard",
        )
        assert payload.corpus_text == "<corpus>test</corpus>"
        assert payload.cache_name is None

    def test_cache_payload(self):
        payload = AgentContextPayload(
            cache_name="cachedContents/abc123",
            strategy="high_volume_cache",
        )
        assert payload.corpus_text is None
        assert payload.cache_name == "cachedContents/abc123"


class TestContextManager:
    """Tests du gestionnaire de contexte."""

    def test_init_standard_strategy(self):
        cm = ContextManager(GenerationStrategy.STANDARD)
        assert cm.strategy == GenerationStrategy.STANDARD
        assert cm.get_cache_id() is None

    def test_format_corpus_xml_empty(self):
        """Corpus vide → XML racine vide."""
        xml = ContextManager._format_corpus_xml(None)
        assert "<corpus" in xml

    def test_format_corpus_xml_with_chunks(self):
        """Corpus avec chunks → XML structuré."""

        class MockChunk:
            def __init__(self, text, source_file, page_number=1, section_title=""):
                self.text = text
                self.source_file = source_file
                self.page_number = page_number
                self.section_title = section_title

        class MockCorpus:
            def __init__(self, chunks):
                self.chunks = chunks

        corpus = MockCorpus([
            MockChunk("Premier paragraphe", "doc1.pdf", 1, "Intro"),
            MockChunk("Deuxième paragraphe", "doc1.pdf", 2, "Corps"),
            MockChunk("Texte autre doc", "doc2.pdf", 1, "Résumé"),
        ])

        xml = ContextManager._format_corpus_xml(corpus)
        assert "<corpus>" in xml
        assert "<document" in xml
        assert 'source="doc1.pdf"' in xml
        assert 'source="doc2.pdf"' in xml
        assert "Premier paragraphe" in xml
        assert "Texte autre doc" in xml

    def test_get_context_for_agent_standard(self):
        """Mode STANDARD → corpus_text rempli, cache_name None."""
        cm = ContextManager(GenerationStrategy.STANDARD)
        cm._corpus_xml = "<corpus><doc>Test</doc></corpus>"

        payload = cm.get_context_for_agent(section_id="s1")
        assert payload.corpus_text == "<corpus><doc>Test</doc></corpus>"
        assert payload.cache_name is None
        assert payload.strategy == "standard"

    def test_get_context_for_agent_cache(self):
        """Mode CACHE → corpus_text None, cache_name rempli."""
        cm = ContextManager(GenerationStrategy.HIGH_VOLUME_CACHE)
        cm._cache_name = "cachedContents/xyz789"

        payload = cm.get_context_for_agent(section_id="s2")
        assert payload.corpus_text is None
        assert payload.cache_name == "cachedContents/xyz789"
        assert payload.strategy == "high_volume_cache"

    def test_get_cache_id_none(self):
        """get_cache_id retourne None si aucun cache créé."""
        cm = ContextManager(GenerationStrategy.STANDARD)
        assert cm.get_cache_id() is None

    def test_get_cache_lifecycle(self):
        """get_cache_lifecycle retourne les infos de cycle de vie."""
        cm = ContextManager(GenerationStrategy.STANDARD)
        lifecycle = cm.get_cache_lifecycle()
        assert lifecycle["cache_name"] is None
        assert lifecycle["renewal_count"] == 0
        assert lifecycle["heartbeat_active"] is False

    @pytest.mark.asyncio
    async def test_prepare_standard(self):
        """prepare() en mode STANDARD stocke le XML en mémoire."""

        class MockCorpus:
            chunks = []

        cm = ContextManager(GenerationStrategy.STANDARD)
        await cm.prepare(MockCorpus(), system_instruction="Test instruction")

        assert cm._corpus_xml is not None
        assert cm.get_cache_id() is None

    @pytest.mark.asyncio
    async def test_cleanup_no_cache(self):
        """cleanup() sans cache ne lève pas d'erreur."""
        cm = ContextManager(GenerationStrategy.STANDARD)
        await cm.cleanup()  # Should not raise

    def test_format_plan_xml(self):
        """_format_plan_xml génère du XML structuré."""

        class MockSection:
            def __init__(self, id, title, level, description=""):
                self.id = id
                self.title = title
                self.level = level
                self.description = description

        class MockPlan:
            title = "Mon plan"
            objective = "Tester le XML"
            sections = [
                MockSection("s1", "Introduction", 1, "Desc intro"),
                MockSection("s2", "Analyse", 2),
            ]

            def to_dict(self):
                return {}

        xml = ContextManager._format_plan_xml(MockPlan())
        assert "<plan" in xml
        assert 'title="Mon plan"' in xml
        assert '<section' in xml
        assert 'id="s1"' in xml
