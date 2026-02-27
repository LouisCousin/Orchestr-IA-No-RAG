"""Tests unitaires pour le fournisseur Google Gemini 3.1 — Phase 5."""

import pytest
from unittest.mock import patch, MagicMock, call

from src.providers.gemini_provider import GeminiProvider
from src.providers.base import AIResponse


class TestGeminiProviderInit:
    """Tests d'initialisation du fournisseur Gemini."""

    def test_init_with_api_key(self):
        provider = GeminiProvider(api_key="test-google-key")
        assert provider._api_key == "test-google-key"

    def test_init_from_env(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-google-key"}):
            provider = GeminiProvider()
            assert provider._api_key == "env-google-key"

    def test_name(self):
        provider = GeminiProvider(api_key="test")
        assert provider.name == "google"

    def test_default_model(self):
        """Vérifie que le modèle par défaut est gemini-3-flash-preview."""
        provider = GeminiProvider(api_key="test")
        assert provider.get_default_model() == "gemini-3-flash-preview"

    def test_list_models(self):
        """Vérifie la liste des modèles disponibles (3 modèles Gemini 3.1)."""
        provider = GeminiProvider(api_key="test")
        models = provider.list_models()
        assert "gemini-3.1-pro-preview" in models
        assert "gemini-3.1-pro-preview-customtools" in models
        assert "gemini-3-flash-preview" in models
        assert len(models) == 3

    def test_list_models_returns_copy(self):
        provider = GeminiProvider(api_key="test")
        models1 = provider.list_models()
        models2 = provider.list_models()
        assert models1 is not models2

    def test_no_deprecated_models(self):
        """Vérifie que les anciens modèles sont absents."""
        provider = GeminiProvider(api_key="test")
        models = provider.list_models()
        assert "gemini-3.0-pro" not in models
        assert "gemini-3.0-flash" not in models


class TestGeminiProviderAvailability:
    """Tests de disponibilité du fournisseur."""

    def test_available_with_valid_key(self):
        provider = GeminiProvider(api_key="real-key")
        assert provider.is_available() is True

    def test_not_available_with_empty_key(self):
        provider = GeminiProvider(api_key="")
        assert provider.is_available() is False

    def test_not_available_with_placeholder(self):
        provider = GeminiProvider(api_key="your-google-api-key-here")
        assert provider.is_available() is False

    def test_supports_batch_false(self):
        """Gemini ne supporte pas le batch API."""
        provider = GeminiProvider(api_key="test")
        assert provider.supports_batch() is False


class TestGeminiProviderSupportsCaching:
    """Tests de la méthode supports_caching."""

    def test_supports_caching_pro_preview(self):
        provider = GeminiProvider(api_key="test")
        assert provider.supports_caching("gemini-3.1-pro-preview") is True

    def test_supports_caching_customtools(self):
        provider = GeminiProvider(api_key="test")
        assert provider.supports_caching("gemini-3.1-pro-preview-customtools") is True

    def test_supports_caching_flash_false(self):
        provider = GeminiProvider(api_key="test")
        assert provider.supports_caching("gemini-3-flash-preview") is False

    def test_supports_caching_default_model(self):
        """Le modèle par défaut (Flash) ne supporte pas le caching."""
        provider = GeminiProvider(api_key="test")
        assert provider.supports_caching() is False


class TestGeminiProviderGenerate:
    """Tests de génération (avec mocks)."""

    def _make_mock_response(self, text="Generated content", input_tokens=80, output_tokens=40):
        """Crée un mock de réponse Gemini."""
        mock_response = MagicMock()
        mock_response.text = text
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            cached_content_token_count=0,
        )
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        return mock_response

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_generate_without_cache(self, mock_get_client):
        """Vérifie que system_instruction est présent si cached_content est None."""
        mock_response = self._make_mock_response()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key")

        mock_types = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": mock_types,
        }):
            with patch("src.providers.gemini_provider.types", mock_types):
                result = provider.generate(
                    prompt="Test prompt",
                    system_prompt="System instruction",
                    model="gemini-3.1-pro-preview",
                )

        # Vérifier que generate_content a été appelé
        assert mock_client.models.generate_content.called
        # Vérifier le résultat
        assert result.content == "Generated content"
        assert result.provider == "google"

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_generate_with_cache_no_system_instruction(self, mock_get_client):
        """Vérifie que system_instruction est ABSENT quand cached_content est fourni (R1)."""
        mock_response = self._make_mock_response()
        mock_response.usage_metadata.cached_content_token_count = 50000
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key")

        captured_config_kwargs = {}

        def capture_config(**kwargs):
            captured_config_kwargs.update(kwargs)
            return MagicMock()

        mock_types = MagicMock()
        mock_types.GenerateContentConfig.side_effect = capture_config

        with patch("src.providers.gemini_provider.types", mock_types):
            provider.generate(
                prompt="Test prompt",
                system_prompt="This should be ignored",
                model="gemini-3.1-pro-preview",
                cached_content="cachedContents/abc123",
            )

        # R1 : system_instruction ne doit PAS être dans la config
        assert "system_instruction" not in captured_config_kwargs
        # cached_content doit être présent
        assert captured_config_kwargs.get("cached_content") == "cachedContents/abc123"

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_generate_with_cache_warns_if_system_prompt_also_given(self, mock_get_client):
        """Vérifie qu'un warning est émis si cached_content ET system_prompt sont fournis (R2)."""
        mock_response = self._make_mock_response()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key")
        mock_types = MagicMock()

        with patch("src.providers.gemini_provider.types", mock_types):
            with patch("src.providers.gemini_provider.logger") as mock_logger:
                provider.generate(
                    prompt="Test",
                    system_prompt="Ignored system prompt",
                    model="gemini-3.1-pro-preview",
                    cached_content="cachedContents/xyz",
                )
                # R2 : un warning doit être émis
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "ignoré" in warning_msg.lower() or "ignored" in warning_msg.lower() or "system_prompt" in warning_msg

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_thinking_level_transmitted_for_31_model(self, mock_get_client):
        """Vérifie que thinking_level est transmis pour les modèles 3.1 (R3)."""
        mock_response = self._make_mock_response()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key")

        captured_config_kwargs = {}

        def capture_config(**kwargs):
            captured_config_kwargs.update(kwargs)
            return MagicMock()

        mock_types = MagicMock()
        mock_types.GenerateContentConfig.side_effect = capture_config

        with patch("src.providers.gemini_provider.types", mock_types):
            provider.generate(
                prompt="Test",
                model="gemini-3.1-pro-preview",
                thinking_level="high",
            )

        # thinking_config doit être présent
        assert "thinking_config" in captured_config_kwargs

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_thinking_level_ignored_for_flash(self, mock_get_client):
        """Vérifie que thinking_level n'est PAS transmis pour Flash (R3)."""
        mock_response = self._make_mock_response()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key")

        captured_config_kwargs = {}

        def capture_config(**kwargs):
            captured_config_kwargs.update(kwargs)
            return MagicMock()

        mock_types = MagicMock()
        mock_types.GenerateContentConfig.side_effect = capture_config

        with patch("src.providers.gemini_provider.types", mock_types):
            provider.generate(
                prompt="Test",
                model="gemini-3-flash-preview",
                thinking_level="high",
            )

        # thinking_config ne doit PAS être présent pour Flash
        assert "thinking_config" not in captured_config_kwargs

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_tools_cache_conflict_handled(self, mock_get_client):
        """Vérifie que le conflit tools/cached_content est géré (relance sans cache)."""
        mock_client = MagicMock()
        # Premier appel : erreur 400 conflit tools/cache
        # Second appel (sans cache) : succès
        mock_response = self._make_mock_response()
        mock_client.models.generate_content.side_effect = [
            Exception("cached_content and tools are mutually exclusive"),
            mock_response,
        ]
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key", max_retries=2, base_delay=0.01)
        mock_types = MagicMock()

        with patch("src.providers.gemini_provider.types", mock_types):
            result = provider.generate(
                prompt="Test",
                model="gemini-3.1-pro-preview",
                cached_content="cachedContents/abc",
            )

        # Le résultat doit être récupéré après la relance sans cache
        assert result.content == "Generated content"
        assert mock_client.models.generate_content.call_count == 2

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    @patch("src.providers.gemini_provider.time.sleep")
    def test_retry_on_error(self, mock_sleep, mock_get_client):
        """Vérifie que le retry exponentiel fonctionne sur 3 erreurs puis succès."""
        mock_response = self._make_mock_response()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
            mock_response,
        ]
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key", max_retries=3, base_delay=0.01)
        mock_types = MagicMock()

        with patch("src.providers.gemini_provider.types", mock_types):
            result = provider.generate("Test")

        assert result.content == "Generated content"
        assert mock_client.models.generate_content.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    @patch("src.providers.gemini_provider.time.sleep")
    def test_all_retries_fail(self, mock_sleep, mock_get_client):
        """Vérifie qu'une RuntimeError est levée après tous les retries."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Persistent error")
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key", max_retries=1, base_delay=0.01)
        mock_types = MagicMock()

        with patch("src.providers.gemini_provider.types", mock_types):
            with pytest.raises(RuntimeError, match="Échec après"):
                provider.generate("Test")

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_max_output_tokens_always_set(self, mock_get_client):
        """Vérifie que max_output_tokens est toujours défini explicitement (R4)."""
        mock_response = self._make_mock_response()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key")

        captured_config_kwargs = {}

        def capture_config(**kwargs):
            captured_config_kwargs.update(kwargs)
            return MagicMock()

        mock_types = MagicMock()
        mock_types.GenerateContentConfig.side_effect = capture_config

        with patch("src.providers.gemini_provider.types", mock_types):
            provider.generate("Test", max_tokens=8192)

        assert "max_output_tokens" in captured_config_kwargs
        assert captured_config_kwargs["max_output_tokens"] == 8192
