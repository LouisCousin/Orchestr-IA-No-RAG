"""Tests unitaires pour le CacheHeartbeat — v4.0."""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.core.cache_heartbeat import CacheHeartbeat


class TestCacheHeartbeat:
    """Tests du renouvellement automatique de TTL."""

    def test_init_default_values(self):
        """Valeurs par défaut correctes."""
        mock_manager = MagicMock()
        heartbeat = CacheHeartbeat(mock_manager, "cache_001")
        assert heartbeat.RENEWAL_INTERVAL_SECONDS == 1800
        assert heartbeat.RENEWAL_TTL_EXTENSION_SECONDS == 7200
        assert heartbeat.renewal_count == 0
        assert heartbeat.is_running is False

    def test_init_custom_config(self):
        """Configuration personnalisée depuis config."""
        mock_manager = MagicMock()
        config = {
            "gemini": {
                "heartbeat_interval_seconds": 600,
            }
        }
        heartbeat = CacheHeartbeat(mock_manager, "cache_002", config=config)
        assert heartbeat.RENEWAL_INTERVAL_SECONDS == 600

    def test_stop_without_start(self):
        """stop() sans start() ne lève pas d'erreur."""
        mock_manager = MagicMock()
        heartbeat = CacheHeartbeat(mock_manager, "cache_003")
        heartbeat.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Démarrage et arrêt du heartbeat."""
        mock_manager = MagicMock()
        heartbeat = CacheHeartbeat(mock_manager, "cache_004")

        # Réduire l'intervalle pour le test
        heartbeat.RENEWAL_INTERVAL_SECONDS = 0.05

        heartbeat.start()
        assert heartbeat.is_running is True

        # Laisser tourner un petit moment
        await asyncio.sleep(0.15)

        heartbeat.stop()
        assert heartbeat.renewal_count >= 1

    @pytest.mark.asyncio
    async def test_double_start_ignored(self):
        """Un double start() est ignoré."""
        mock_manager = MagicMock()
        heartbeat = CacheHeartbeat(mock_manager, "cache_005")
        heartbeat.RENEWAL_INTERVAL_SECONDS = 100  # Long interval

        heartbeat.start()
        first_task = heartbeat._task

        heartbeat.start()  # Should be ignored
        assert heartbeat._task is first_task

        heartbeat.stop()
