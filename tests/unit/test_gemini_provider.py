"""Tests unitaires pour le fournisseur Google Gemini."""

import pytest
from unittest.mock import patch, MagicMock

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
        provider = GeminiProvider(api_key="test")
        assert provider.get_default_model() == "gemini-3.0-flash"

    def test_list_models(self):
        provider = GeminiProvider(api_key="test")
        models = provider.list_models()
        assert "gemini-3.0-pro" in models
        assert "gemini-3.0-flash" in models

    def test_list_models_returns_copy(self):
        provider = GeminiProvider(api_key="test")
        models1 = provider.list_models()
        models2 = provider.list_models()
        assert models1 is not models2


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


class TestGeminiProviderGenerate:
    """Tests de génération (avec mocks)."""

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_generate_success(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.text = "Contenu Gemini"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=80,
            candidates_token_count=40,
        )
        mock_response.candidates = [MagicMock(finish_reason="STOP")]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key")

        # Mock le module google.genai.types
        with patch("src.providers.gemini_provider.GeminiProvider.generate") as mock_gen:
            mock_gen.return_value = AIResponse(
                content="Contenu Gemini",
                model="gemini-3.0-flash",
                provider="google",
                input_tokens=80,
                output_tokens=40,
                total_tokens=120,
                finish_reason="STOP",
            )
            result = provider.generate("Test prompt")

        assert isinstance(result, AIResponse)
        assert result.content == "Contenu Gemini"
        assert result.provider == "google"

    @patch("src.providers.gemini_provider.GeminiProvider._get_client")
    def test_generate_all_retries_fail(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        provider = GeminiProvider(api_key="test-key", max_retries=1, base_delay=0.01)

        # Simuler l'import de types
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock(), "google.genai.types": MagicMock()}):
            with pytest.raises(RuntimeError, match="Échec après"):
                provider.generate("Test")
