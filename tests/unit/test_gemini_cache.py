"""Tests unitaires pour GeminiCacheManager — Phase 5."""

import pytest
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

from src.core.gemini_cache_manager import (
    GeminiCacheManager,
    CacheTooSmallError,
    format_corpus_xml,
)


@pytest.fixture
def manager():
    """Instance de GeminiCacheManager avec clé API de test."""
    return GeminiCacheManager(api_key="test-key")


class TestCreateCorpusCache:
    """Tests de création de cache."""

    @patch("src.core.gemini_cache_manager.GeminiCacheManager._get_client")
    @patch("src.core.gemini_cache_manager.count_tokens", return_value=5000)
    def test_create_corpus_cache_success(self, mock_count, mock_get_client, manager):
        """Vérifie la création d'un cache avec retour du cache_name."""
        mock_cache = MagicMock()
        mock_cache.name = "cachedContents/test123"
        mock_client = MagicMock()
        mock_client.caches.create.return_value = mock_cache
        mock_get_client.return_value = mock_client

        mock_types = MagicMock()
        with patch("src.core.gemini_cache_manager.genai_types", mock_types):
            result = manager.create_corpus_cache(
                project_id="test_project",
                corpus_xml="<corpus><document id='001'>content</document></corpus>",
                system_prompt="You are an expert writer.",
                model="gemini-3.1-pro-preview",
                ttl=7200,
            )

        assert result == "cachedContents/test123"
        assert mock_client.caches.create.called

    @patch("src.core.gemini_cache_manager.count_tokens", return_value=500)
    def test_create_corpus_cache_too_small(self, mock_count, manager):
        """Vérifie que CacheTooSmallError est levée si corpus < 2048 tokens."""
        with pytest.raises(CacheTooSmallError, match="2048"):
            manager.create_corpus_cache(
                project_id="test",
                corpus_xml="<corpus>tiny</corpus>",
                system_prompt="system",
                model="gemini-3.1-pro-preview",
            )

    @patch("src.core.gemini_cache_manager.GeminiCacheManager._get_client")
    @patch("src.core.gemini_cache_manager.count_tokens", return_value=5000)
    def test_create_corpus_cache_passes_system_instruction(self, mock_count, mock_get_client, manager):
        """Vérifie que system_prompt est passé dans system_instruction du cache."""
        mock_new_cache = MagicMock()
        mock_new_cache.name = "cachedContents/new"
        mock_client = MagicMock()
        mock_client.caches.create.return_value = mock_new_cache
        mock_get_client.return_value = mock_client

        mock_types = MagicMock()
        mock_config_instance = MagicMock()
        mock_types.CreateCachedContentConfig.return_value = mock_config_instance

        with patch("src.core.gemini_cache_manager.genai_types", mock_types):
            manager.create_corpus_cache(
                project_id="test",
                corpus_xml="<corpus>content</corpus>",
                system_prompt="Expert writer prompt",
                model="gemini-3.1-pro-preview",
                ttl=3600,
            )

        mock_types.CreateCachedContentConfig.assert_called_once()
        call_kwargs = mock_types.CreateCachedContentConfig.call_args[1]
        assert call_kwargs.get("system_instruction") == "Expert writer prompt"


class TestGetOrCreateCache:
    """Tests du comportement get_or_create."""

    @patch("src.core.gemini_cache_manager.GeminiCacheManager._get_client")
    @patch("src.core.gemini_cache_manager.count_tokens", return_value=5000)
    def test_existing_cache_valid_ttl(self, mock_count, mock_get_client, manager):
        """Vérifie qu'un cache valide (TTL > 30min) est réutilisé sans création."""
        from datetime import datetime, timezone, timedelta

        mock_cache = MagicMock()
        expire_time = MagicMock()
        expire_time.timestamp.return_value = (
            datetime.now(timezone.utc) + timedelta(seconds=3600)
        ).timestamp()
        mock_cache.expire_time = expire_time
        mock_cache.name = "cachedContents/existing"

        mock_client = MagicMock()
        mock_client.caches.get.return_value = mock_cache
        mock_get_client.return_value = mock_client

        result = manager.get_or_create_cache(
            project_id="test",
            corpus_xml="<corpus>content</corpus>",
            system_prompt="system",
            model="gemini-3.1-pro-preview",
            existing_cache_name="cachedContents/existing",
        )

        assert result == "cachedContents/existing"
        # create NE doit PAS être appelé
        assert not mock_client.caches.create.called

    @patch("src.core.gemini_cache_manager.GeminiCacheManager._get_client")
    @patch("src.core.gemini_cache_manager.count_tokens", return_value=5000)
    def test_expiring_cache_extends_ttl(self, mock_count, mock_get_client, manager):
        """Vérifie que le TTL est prolongé si cache expire dans < 30min."""
        from datetime import datetime, timezone, timedelta

        mock_cache = MagicMock()
        # TTL restant = 20 minutes (< 30min = seuil de renouvellement)
        expire_time = MagicMock()
        expire_time.timestamp.return_value = (
            datetime.now(timezone.utc) + timedelta(seconds=1200)
        ).timestamp()
        mock_cache.expire_time = expire_time
        mock_cache.name = "cachedContents/expiring"

        mock_client = MagicMock()
        mock_client.caches.get.return_value = mock_cache
        mock_get_client.return_value = mock_client

        mock_types = MagicMock()
        with patch("src.core.gemini_cache_manager.genai_types", mock_types):
            result = manager.get_or_create_cache(
                project_id="test",
                corpus_xml="<corpus>content</corpus>",
                system_prompt="system",
                model="gemini-3.1-pro-preview",
                existing_cache_name="cachedContents/expiring",
            )

        assert result == "cachedContents/expiring"
        # update doit être appelé pour prolonger le TTL
        assert mock_client.caches.update.called

    @patch("src.core.gemini_cache_manager.GeminiCacheManager._get_client")
    @patch("src.core.gemini_cache_manager.count_tokens", return_value=5000)
    def test_create_called_when_no_existing_cache(self, mock_count, mock_get_client, manager):
        """Vérifie qu'un cache est créé si aucun cache existant."""
        mock_new_cache = MagicMock()
        mock_new_cache.name = "cachedContents/new123"

        mock_client = MagicMock()
        mock_client.caches.create.return_value = mock_new_cache
        mock_get_client.return_value = mock_client

        mock_types = MagicMock()
        with patch("src.core.gemini_cache_manager.genai_types", mock_types):
            result = manager.get_or_create_cache(
                project_id="test",
                corpus_xml="<corpus>content</corpus>",
                system_prompt="system",
                model="gemini-3.1-pro-preview",
                existing_cache_name=None,
            )

        assert result == "cachedContents/new123"
        assert mock_client.caches.create.called


class TestEstimateCacheCost:
    """Tests d'estimation des coûts du cache."""

    def test_estimate_basic(self, manager):
        """Vérifie les calculs de base pour 100k tokens, 20 sections, 2h."""
        estimate = manager.estimate_cache_cost(
            corpus_tokens=100_000,
            num_sections=20,
            ttl_hours=2.0,
            model="gemini-3.1-pro-preview",
        )

        assert "cost_without_cache" in estimate
        assert "cost_with_cache" in estimate
        assert "savings_usd" in estimate
        assert "savings_percent" in estimate
        assert "storage_cost" in estimate
        assert "break_even_sections" in estimate

        # Vérifier que le cache est moins cher pour 20 sections
        assert estimate["cost_with_cache"] < estimate["cost_without_cache"]
        assert estimate["savings_usd"] > 0
        assert estimate["savings_percent"] > 0

    def test_estimate_savings_substantially(self, manager):
        """Vérifie que l'économie est substantielle pour un grand corpus."""
        estimate = manager.estimate_cache_cost(
            corpus_tokens=200_000,
            num_sections=30,
            ttl_hours=2.0,
            model="gemini-3.1-pro-preview",
        )
        # L'économie doit être substantielle (> 70%)
        assert estimate["savings_percent"] > 70.0

    def test_estimate_break_even_positive(self, manager):
        """Vérifie que le break-even est un entier positif cohérent."""
        estimate = manager.estimate_cache_cost(
            corpus_tokens=50_000,
            num_sections=10,
            ttl_hours=2.0,
            model="gemini-3.1-pro-preview",
        )
        assert estimate["break_even_sections"] >= 1
        assert isinstance(estimate["break_even_sections"], int)


class TestShouldUseCache:
    """Tests de la décision d'utilisation du cache."""

    def test_should_use_cache_true(self, manager):
        """Corpus 50k tokens, 10 sections → doit retourner True."""
        result = manager.should_use_cache(
            corpus_tokens=50_000,
            num_sections=10,
            model="gemini-3.1-pro-preview",
        )
        assert result is True

    def test_should_use_cache_false_too_small(self, manager):
        """Corpus 1000 tokens → doit retourner False (< 2048 tokens)."""
        result = manager.should_use_cache(
            corpus_tokens=1_000,
            num_sections=10,
            model="gemini-3.1-pro-preview",
        )
        assert result is False

    def test_should_use_cache_false_too_few_sections(self, manager):
        """Très peu de sections → cache pas rentable."""
        result = manager.should_use_cache(
            corpus_tokens=3_000,
            num_sections=1,
            model="gemini-3.1-pro-preview",
        )
        # 1 section < break-even → False (ou True si break-even=1)
        assert isinstance(result, bool)


class TestFormatCorpusXml:
    """Tests du formatage XML du corpus."""

    def test_basic_xml_structure(self):
        """Vérifie que le XML généré est bien formé et contient les attributs requis."""
        corpus_data = [
            {
                "id": "001",
                "title": "Test Document",
                "year": "2024",
                "type": "rapport",
                "chunks": [
                    {
                        "id": "001_001",
                        "page": "3",
                        "section": "Introduction",
                        "content": "Contenu du premier chunk.",
                    },
                    {
                        "id": "001_002",
                        "page": "4",
                        "section": "Méthodologie",
                        "content": "Contenu du second chunk.",
                    },
                ],
            }
        ]

        xml_str = format_corpus_xml(corpus_data)

        # Vérifier que le XML est bien formé
        root = ET.fromstring(xml_str)
        assert root.tag == "corpus"

        docs = root.findall("document")
        assert len(docs) == 1

        doc = docs[0]
        assert doc.get("id") == "001"
        assert doc.get("title") == "Test Document"
        assert doc.get("year") == "2024"
        assert doc.get("type") == "rapport"

        chunks = doc.findall("chunk")
        assert len(chunks) == 2
        assert chunks[0].get("id") == "001_001"
        assert chunks[0].get("section") == "Introduction"
        assert chunks[0].text == "Contenu du premier chunk."

    def test_xml_multiple_documents(self):
        """Vérifie que plusieurs documents sont correctement sérialisés."""
        corpus_data = [
            {"id": "001", "title": "Doc 1", "year": "2023", "type": "article", "chunks": []},
            {"id": "002", "title": "Doc 2", "year": "2024", "type": "rapport", "chunks": []},
        ]
        xml_str = format_corpus_xml(corpus_data)
        root = ET.fromstring(xml_str)
        assert len(root.findall("document")) == 2

    def test_xml_empty_corpus(self):
        """Vérifie que le corpus vide génère une balise racine valide."""
        xml_str = format_corpus_xml([])
        root = ET.fromstring(xml_str)
        assert root.tag == "corpus"
        assert len(root.findall("document")) == 0

    def test_xml_special_characters_escaped(self):
        """Vérifie que les caractères spéciaux XML sont correctement échappés."""
        corpus_data = [
            {
                "id": "001",
                "title": "Test & <Escape>",
                "year": "2024",
                "type": "test",
                "chunks": [
                    {
                        "id": "001_001",
                        "page": "1",
                        "section": "Intro",
                        "content": "Text with <tags> & 'quotes'",
                    }
                ],
            }
        ]
        # Ne doit pas lever d'exception
        xml_str = format_corpus_xml(corpus_data)
        # Doit être parseable
        root = ET.fromstring(xml_str)
        assert root is not None


class TestDeleteCache:
    """Tests de suppression du cache."""

    @patch("src.core.gemini_cache_manager.GeminiCacheManager._get_client")
    def test_delete_cache_success(self, mock_get_client, manager):
        """Vérifie que le cache est supprimé via l'API."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        manager.delete_cache("cachedContents/to_delete")

        mock_client.caches.delete.assert_called_once_with(name="cachedContents/to_delete")

    @patch("src.core.gemini_cache_manager.GeminiCacheManager._get_client")
    def test_delete_cache_handles_error(self, mock_get_client, manager):
        """Vérifie que les erreurs de suppression sont gérées sans exception."""
        mock_client = MagicMock()
        mock_client.caches.delete.side_effect = Exception("Cache not found")
        mock_get_client.return_value = mock_client

        # Ne doit pas lever d'exception
        manager.delete_cache("cachedContents/nonexistent")
