"""Renouvellement automatique du TTL des caches Gemini — v4.0.

Maintient le cache vivant pendant toute la durée de la génération
via un renouvellement périodique toutes les 30 minutes.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("orchestria")


class CacheHeartbeat:
    """Tâche asyncio de fond pour renouveler le TTL d'un cache Gemini."""

    RENEWAL_INTERVAL_SECONDS: int = 1800  # 30 minutes
    RENEWAL_TTL_EXTENSION_SECONDS: int = 7200  # +2h à chaque renouvellement

    def __init__(self, cache_manager, cache_name: str, config: dict | None = None):
        self._cache_manager = cache_manager
        self._cache_name = cache_name
        self._task: Optional[asyncio.Task] = None
        self._renewal_count: int = 0

        if config:
            gemini_cfg = config.get("gemini", {})
            self.RENEWAL_INTERVAL_SECONDS = gemini_cfg.get(
                "heartbeat_interval_seconds", self.RENEWAL_INTERVAL_SECONDS
            )

    def start(self) -> None:
        """Démarre la boucle de heartbeat en tâche asyncio de fond."""
        if self._task and not self._task.done():
            logger.warning("[Heartbeat] Déjà en cours, ignoré.")
            return
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            f"[Heartbeat] Démarré pour {self._cache_name} "
            f"(intervalle={self.RENEWAL_INTERVAL_SECONDS}s)"
        )

    def stop(self) -> None:
        """Arrête la boucle de heartbeat."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info(
                f"[Heartbeat] Arrêté pour {self._cache_name} "
                f"({self._renewal_count} renouvellements effectués)"
            )

    @property
    def renewal_count(self) -> int:
        return self._renewal_count

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _heartbeat_loop(self) -> None:
        """Boucle de renouvellement périodique du TTL."""
        while True:
            try:
                await asyncio.sleep(self.RENEWAL_INTERVAL_SECONDS)
                self._cache_manager.extend_cache_ttl(
                    self._cache_name, ttl=self.RENEWAL_TTL_EXTENSION_SECONDS
                )
                self._renewal_count += 1
                logger.info(
                    f"[Heartbeat] TTL renouvelé pour {self._cache_name} "
                    f"(+{self.RENEWAL_TTL_EXTENSION_SECONDS}s, "
                    f"total renouvellements: {self._renewal_count})"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Heartbeat] Échec renouvellement TTL: {e}")
