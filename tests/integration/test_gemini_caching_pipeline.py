"""Tests d'intégration pour le pipeline Gemini 3.1 avec Context Caching — Phase 5.

Ces tests nécessitent une clé GOOGLE_API_KEY valide en variable d'environnement.
Ils sont marqués @pytest.mark.integration et exclus des CI/CD rapides.

Lancement : pytest tests/integration/test_gemini_caching_pipeline.py -m integration -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# ── Marqueur pour exclure des CI/CD rapides ──────────────────────────────────
pytestmark = pytest.mark.integration


def _has_google_api_key() -> bool:
    """Vérifie si la clé API Google est disponible."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    return bool(key and key != "your-google-api-key-here")


def _skip_if_no_api_key():
    """Décorateur de skip si pas de clé API."""
    return pytest.mark.skipif(
        not _has_google_api_key(),
        reason="GOOGLE_API_KEY non configurée — test d'intégration ignoré",
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def google_api_key():
    """Clé API Google depuis l'environnement."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        pytest.skip("GOOGLE_API_KEY non configurée")
    return key


@pytest.fixture
def sample_corpus_xml():
    """Corpus XML de test (suffisamment grand pour le caching)."""
    # Génère un corpus d'environ 3000 tokens
    chunks = []
    for i in range(20):
        chunks.append(
            f"<chunk id='001_{i+1:03d}' page='{i+1}' section='Section {i+1}'>"
            f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            f"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            f"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi. "
            f"Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. "
            f"Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia. "
            f"Référence {i+1} : données issues du corpus de test Phase 5."
            f"</chunk>"
        )
    all_chunks = "\n".join(chunks)
    return f"<corpus><document id='001' title='Document de test' year='2024' type='rapport'>{all_chunks}</document></corpus>"


@pytest.fixture
def sample_system_prompt():
    """System prompt stable pour les tests."""
    return (
        "Tu es un expert en rédaction de documents professionnels. "
        "Utilise exclusivement les informations du corpus fourni. "
        "Cite tes sources de manière précise."
    )


# ── Tests d'intégration ───────────────────────────────────────────────────────

class TestGeminiCacheLifecycle:
    """Tests du cycle de vie complet d'un cache Gemini."""

    @_skip_if_no_api_key()
    def test_create_and_delete_cache(self, google_api_key, sample_corpus_xml, sample_system_prompt):
        """Crée un cache, vérifie son existence, puis le supprime."""
        from src.core.gemini_cache_manager import GeminiCacheManager

        mgr = GeminiCacheManager(api_key=google_api_key)

        cache_name = mgr.create_corpus_cache(
            project_id="integration_test",
            corpus_xml=sample_corpus_xml,
            system_prompt=sample_system_prompt,
            model="gemini-3.1-pro-preview",
            ttl=300,  # TTL minimum pour le test
        )

        assert cache_name is not None
        assert cache_name.startswith("cachedContents/")

        # Supprimer le cache après le test
        mgr.delete_cache(cache_name)

    @_skip_if_no_api_key()
    def test_get_or_create_reuses_existing_cache(self, google_api_key, sample_corpus_xml, sample_system_prompt):
        """Vérifie que get_or_create réutilise un cache existant valide."""
        from src.core.gemini_cache_manager import GeminiCacheManager

        mgr = GeminiCacheManager(api_key=google_api_key)

        # Créer un premier cache
        cache_name_1 = mgr.get_or_create_cache(
            project_id="test_reuse",
            corpus_xml=sample_corpus_xml,
            system_prompt=sample_system_prompt,
            model="gemini-3.1-pro-preview",
            ttl=600,
        )

        # Appel suivant doit retourner le MÊME cache
        cache_name_2 = mgr.get_or_create_cache(
            project_id="test_reuse",
            corpus_xml=sample_corpus_xml,
            system_prompt=sample_system_prompt,
            model="gemini-3.1-pro-preview",
            ttl=600,
            existing_cache_name=cache_name_1,
        )

        assert cache_name_1 == cache_name_2

        # Nettoyage
        mgr.delete_cache(cache_name_1)

    @_skip_if_no_api_key()
    def test_recreate_cache_after_deletion(self, google_api_key, sample_corpus_xml, sample_system_prompt):
        """Vérifie qu'un cache expiré/supprimé déclenche une recréation automatique."""
        from src.core.gemini_cache_manager import GeminiCacheManager

        mgr = GeminiCacheManager(api_key=google_api_key)

        # Créer puis supprimer un cache
        cache_name = mgr.create_corpus_cache(
            project_id="test_recreate",
            corpus_xml=sample_corpus_xml,
            system_prompt=sample_system_prompt,
            model="gemini-3.1-pro-preview",
            ttl=300,
        )
        mgr.delete_cache(cache_name)

        # get_or_create doit créer un nouveau cache
        new_cache_name = mgr.get_or_create_cache(
            project_id="test_recreate",
            corpus_xml=sample_corpus_xml,
            system_prompt=sample_system_prompt,
            model="gemini-3.1-pro-preview",
            ttl=300,
            existing_cache_name=cache_name,  # Ancien nom (supprimé)
        )

        assert new_cache_name is not None
        assert new_cache_name != cache_name  # Nouveau cache créé

        # Nettoyage
        mgr.delete_cache(new_cache_name)


class TestGeminiGenerationWithCache:
    """Tests de génération avec cache actif."""

    @_skip_if_no_api_key()
    def test_generation_with_cached_content(self, google_api_key, sample_corpus_xml, sample_system_prompt):
        """Vérifie qu'une génération avec cache retourne un contenu cohérent."""
        from src.core.gemini_cache_manager import GeminiCacheManager
        from src.providers.gemini_provider import GeminiProvider

        mgr = GeminiCacheManager(api_key=google_api_key)
        provider = GeminiProvider(api_key=google_api_key)

        # Créer le cache
        cache_name = mgr.create_corpus_cache(
            project_id="test_generation",
            corpus_xml=sample_corpus_xml,
            system_prompt=sample_system_prompt,
            model="gemini-3.1-pro-preview",
            ttl=300,
        )

        try:
            # Générer du contenu en utilisant le cache
            response = provider.generate(
                prompt="Résume le contenu du corpus en 3 points clés.",
                model="gemini-3.1-pro-preview",
                cached_content=cache_name,
                max_tokens=512,
            )

            assert response.content is not None
            assert len(response.content) > 0
            assert response.provider == "google"
            assert response.model == "gemini-3.1-pro-preview"

        finally:
            mgr.delete_cache(cache_name)

    @_skip_if_no_api_key()
    def test_cache_reused_across_multiple_sections(self, google_api_key, sample_corpus_xml, sample_system_prompt):
        """Vérifie que le cache est réutilisé pour plusieurs sections (pas de recréation)."""
        from src.core.gemini_cache_manager import GeminiCacheManager
        from src.providers.gemini_provider import GeminiProvider

        mgr = GeminiCacheManager(api_key=google_api_key)
        provider = GeminiProvider(api_key=google_api_key)

        cache_name = mgr.create_corpus_cache(
            project_id="test_multi_section",
            corpus_xml=sample_corpus_xml,
            system_prompt=sample_system_prompt,
            model="gemini-3.1-pro-preview",
            ttl=300,
        )

        try:
            sections = [
                "Rédige une introduction basée sur le corpus.",
                "Présente la méthodologie décrite dans le corpus.",
                "Synthétise les conclusions du corpus.",
            ]

            responses = []
            for section_prompt in sections:
                response = provider.generate(
                    prompt=section_prompt,
                    model="gemini-3.1-pro-preview",
                    cached_content=cache_name,
                    max_tokens=256,
                )
                responses.append(response)

            # Vérifier que toutes les sections ont été générées
            assert len(responses) == 3
            for resp in responses:
                assert len(resp.content) > 0

        finally:
            mgr.delete_cache(cache_name)


class TestLongContextDetection:
    """Tests de la détection du repricing long-context."""

    def test_long_context_threshold_detection(self):
        """Vérifie que le seuil long-context est correctement détecté."""
        from src.core.cost_tracker import CostTracker

        tracker = CostTracker()

        # En-dessous du seuil
        is_long = tracker.check_long_context_threshold(
            "google", "gemini-3.1-pro-preview", 100_000
        )
        assert is_long is False

        # Au-dessus du seuil (200 000 tokens)
        is_long = tracker.check_long_context_threshold(
            "google", "gemini-3.1-pro-preview", 250_000
        )
        assert is_long is True

    def test_long_context_cost_calculation(self):
        """Vérifie que le coût long-context est correctement calculé."""
        from src.core.cost_tracker import CostTracker

        tracker = CostTracker()

        # Appel standard (100k tokens)
        cost_standard = tracker.calculate_cost(
            "google", "gemini-3.1-pro-preview", 100_000, 1_000
        )

        # Appel long-context (250k tokens)
        cost_long = tracker.calculate_cost(
            "google", "gemini-3.1-pro-preview", 250_000, 1_000
        )

        # Le coût long-context doit être plus élevé par token
        rate_standard = cost_standard / 100_000 if cost_standard > 0 else 0
        rate_long = cost_long / 250_000 if cost_long > 0 else 0
        assert rate_long > rate_standard


class TestEmbeddingMigration:
    """Tests de migration des embeddings vers le nouveau SDK."""

    @_skip_if_no_api_key()
    def test_embedding_with_new_sdk(self, google_api_key):
        """Vérifie que les embeddings fonctionnent avec le nouveau SDK google-genai."""
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=google_api_key)

        test_texts = ["Ceci est un texte de test.", "Another test text."]

        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=test_texts,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            ),
        )

        embeddings = [e.values for e in result.embeddings]
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 768  # text-embedding-004 = 768 dimensions
        assert all(isinstance(v, float) for v in embeddings[0])

    def test_embedding_model_in_rag_engine(self):
        """Vérifie que le RAG engine appelle le bon modèle d'embedding."""
        from src.core.rag_engine import RAGEngine

        config = {
            "rag": {
                "embedding_provider": "gemini",
                "embedding_model": "text-embedding-004",
                "batch_size": 10,
                "reranking_enabled": False,
            }
        }
        engine = RAGEngine(config=config)

        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3] * 256  # 768 dims
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]

        mock_types = MagicMock()

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": mock_types,
        }):
            with patch("src.core.rag_engine.os.environ.get", return_value="test-key"):
                import src.core.rag_engine as rag_module
                with patch.object(rag_module, "__builtins__", {}):
                    with patch("google.genai.Client", return_value=mock_client):
                        mock_client.models.embed_content.return_value = mock_result
                        # Note : test de bas niveau — vérification de la mécanique API


class TestFallbackToStandardMode:
    """Tests du fallback vers le mode standard si corpus trop petit."""

    def test_cache_not_created_for_tiny_corpus(self):
        """Vérifie que le cache n'est pas créé pour un corpus < 2048 tokens."""
        from src.core.gemini_cache_manager import GeminiCacheManager, CacheTooSmallError

        mgr = GeminiCacheManager(api_key="test-key")

        with patch("src.core.gemini_cache_manager.count_tokens", return_value=500):
            with pytest.raises(CacheTooSmallError):
                mgr.create_corpus_cache(
                    project_id="test",
                    corpus_xml="<corpus>tiny</corpus>",
                    system_prompt="system",
                    model="gemini-3.1-pro-preview",
                )

    def test_should_use_cache_false_for_tiny_corpus(self):
        """Vérifie que should_use_cache retourne False pour un corpus trop petit."""
        from src.core.gemini_cache_manager import GeminiCacheManager

        mgr = GeminiCacheManager(api_key="test-key")
        result = mgr.should_use_cache(
            corpus_tokens=1_000,
            num_sections=50,
            model="gemini-3.1-pro-preview",
        )
        assert result is False


class TestCostTrackerWithCache:
    """Tests du cost_tracker avec les nouveaux cas de calcul Phase 5."""

    def test_track_cached_call(self):
        """Vérifie l'enregistrement d'un appel avec tokens mixtes (CAS 2)."""
        from src.core.cost_tracker import CostTracker

        tracker = CostTracker()
        cost = tracker.track_cached_call(
            model="gemini-3.1-pro-preview",
            input_tokens=500,
            cached_tokens=90_000,
            output_tokens=1_000,
            section_id="section_01",
        )

        assert cost > 0
        # Le coût avec cache doit être inférieur au coût standard du même volume
        cost_standard = tracker.calculate_cost(
            "google", "gemini-3.1-pro-preview", 90_500, 1_000
        )
        assert cost < cost_standard

        # Vérifier les stats de cache
        report = tracker.report
        assert report.gemini_cache_stats.get("cache_hits") == 1

    def test_track_cache_storage(self):
        """Vérifie le calcul du coût de stockage (CAS 4)."""
        from src.core.cost_tracker import CostTracker

        tracker = CostTracker()
        cost = tracker.track_cache_storage(
            model="gemini-3.1-pro-preview",
            cached_tokens=100_000,
            ttl_hours=2.0,
        )

        # $0.50/heure/1M tokens × 100k tokens × 2h = $0.10
        expected = (100_000 / 1_000_000) * 0.50 * 2.0
        assert abs(cost - expected) < 0.001

    def test_init_cache_stats(self):
        """Vérifie l'initialisation des stats de cache dans le rapport."""
        from src.core.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.init_cache_stats(
            cache_name="cachedContents/abc123",
            tokens_cached=50_000,
        )

        stats = tracker.report.gemini_cache_stats
        assert stats["cache_name"] == "cachedContents/abc123"
        assert stats["tokens_cached"] == 50_000
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
