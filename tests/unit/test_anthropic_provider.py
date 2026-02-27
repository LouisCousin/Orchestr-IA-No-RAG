"""Tests unitaires pour le fournisseur Anthropic."""

import pytest
from unittest.mock import patch, MagicMock

from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import AIResponse


class TestAnthropicProviderInit:
    """Tests d'initialisation du fournisseur Anthropic."""

    def test_init_with_api_key(self):
        provider = AnthropicProvider(api_key="sk-ant-test-key")
        assert provider._api_key == "sk-ant-test-key"

    def test_init_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            provider = AnthropicProvider()
            assert provider._api_key == "sk-ant-env-key"

    def test_name(self):
        provider = AnthropicProvider(api_key="test")
        assert provider.name == "anthropic"

    def test_default_model(self):
        provider = AnthropicProvider(api_key="test")
        assert provider.get_default_model() == "claude-sonnet-4-5-20250514"

    def test_list_models(self):
        provider = AnthropicProvider(api_key="test")
        models = provider.list_models()
        assert "claude-opus-4-6" in models
        assert "claude-sonnet-4-5-20250514" in models
        assert "claude-3-5-haiku-20241022" in models

    def test_list_models_returns_copy(self):
        provider = AnthropicProvider(api_key="test")
        models1 = provider.list_models()
        models2 = provider.list_models()
        assert models1 == models2
        assert models1 is not models2


class TestAnthropicProviderAvailability:
    """Tests de disponibilité du fournisseur."""

    def test_available_with_valid_key(self):
        provider = AnthropicProvider(api_key="sk-ant-real-key-here")
        assert provider.is_available() is True

    def test_not_available_with_empty_key(self):
        provider = AnthropicProvider(api_key="")
        assert provider.is_available() is False

    def test_not_available_with_placeholder(self):
        provider = AnthropicProvider(api_key="sk-ant-your-anthropic-api-key-here")
        assert provider.is_available() is False


class TestAnthropicProviderGenerate:
    """Tests de génération (avec mocks)."""

    @patch("src.providers.anthropic_provider.AnthropicProvider._get_client")
    def test_generate_success(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Contenu généré")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_response.stop_reason = "end_turn"
        mock_response.model_dump.return_value = {}

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        result = provider.generate("Test prompt", system_prompt="System")

        assert isinstance(result, AIResponse)
        assert result.content == "Contenu généré"
        assert result.provider == "anthropic"
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @patch("src.providers.anthropic_provider.AnthropicProvider._get_client")
    def test_generate_with_retry(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            Exception("Rate limited"),
            MagicMock(
                content=[MagicMock(type="text", text="OK")],
                usage=MagicMock(input_tokens=10, output_tokens=5),
                stop_reason="end_turn",
                model_dump=MagicMock(return_value={}),
            ),
        ]
        mock_get_client.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test", base_delay=0.01)
        result = provider.generate("Test")

        assert result.content == "OK"
        assert mock_client.messages.create.call_count == 2

    @patch("src.providers.anthropic_provider.AnthropicProvider._get_client")
    def test_generate_all_retries_fail(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Persistent error")
        mock_get_client.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test", max_retries=1, base_delay=0.01)
        with pytest.raises(RuntimeError, match="Échec après"):
            provider.generate("Test")
