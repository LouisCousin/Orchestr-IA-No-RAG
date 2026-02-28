"""Fournisseur Anthropic (Claude) pour Orchestr'IA.

Phase 2.5 : ajout du support batch via Message Batches API.
"""

import os
import random
import time
import logging
from typing import Optional

from src.providers.base import (
    AIResponse, BaseProvider, BatchRequest, BatchStatus, BatchStatusEnum, BatchError,
)

logger = logging.getLogger("orchestria")


class AnthropicProvider(BaseProvider):
    """Fournisseur Anthropic (Claude) avec support batch."""

    MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-5-20250514",
        "claude-3-5-haiku-20241022",
    ]

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 2.0):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "anthropic"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """Génère du contenu via l'API Anthropic avec retry automatique."""
        model = model or self.get_default_model()

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()

                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = client.messages.create(**kwargs)

                content = ""
                for block in (response.content or []):
                    if block.type == "text":
                        content += block.text

                input_tokens = response.usage.input_tokens if response.usage else 0
                output_tokens = response.usage.output_tokens if response.usage else 0

                return AIResponse(
                    content=content,
                    model=model,
                    provider=self.name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    finish_reason=response.stop_reason or "",
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    delay += jitter
                    logger.warning(
                        f"Erreur API Anthropic (tentative {attempt + 1}/{self._max_retries + 1}): {e}. "
                        f"Retry dans {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Échec définitif API Anthropic après {self._max_retries + 1} tentatives: {last_error}")

        raise RuntimeError(f"Échec après {self._max_retries + 1} tentatives: {last_error}")

    def is_available(self) -> bool:
        """Vérifie si la clé API est configurée."""
        return bool(self._api_key and self._api_key != "sk-ant-your-anthropic-api-key-here")

    def get_default_model(self) -> str:
        return "claude-sonnet-4-5-20250514"

    def list_models(self) -> list[str]:
        return self.MODELS.copy()

    # ── Phase 2.5 : Support batch Anthropic (Message Batches API) ──

    def supports_batch(self) -> bool:
        return True

    def submit_batch(self, requests: list[BatchRequest]) -> str:
        """Soumet un batch via l'API Anthropic Message Batches.

        Returns:
            batch_id du batch soumis.
        """
        client = self._get_client()

        batch_requests = []
        for req in requests:
            model = req.model or self.get_default_model()
            params = {
                "model": model,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "messages": [{"role": "user", "content": req.prompt}],
            }
            if req.system_prompt:
                params["system"] = req.system_prompt

            batch_requests.append({
                "custom_id": req.custom_id,
                "params": params,
            })

        batch = client.messages.batches.create(requests=batch_requests)
        logger.info(f"Batch Anthropic soumis : {batch.id} ({len(requests)} requêtes)")
        return batch.id

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """Vérifie le statut d'un batch Anthropic."""
        client = self._get_client()
        status = client.messages.batches.retrieve(batch_id)

        status_map = {
            "in_progress": BatchStatusEnum.IN_PROGRESS,
            "ended": BatchStatusEnum.COMPLETED,
            "canceling": BatchStatusEnum.CANCELLED,
            "canceled": BatchStatusEnum.CANCELLED,
        }

        processing_status = getattr(status, "processing_status", "in_progress")
        batch_status = status_map.get(processing_status, BatchStatusEnum.IN_PROGRESS)

        # Compter les résultats
        counts = getattr(status, "request_counts", None)
        total = 0
        completed = 0
        failed = 0
        if counts:
            total = (
                getattr(counts, "processing", 0)
                + getattr(counts, "succeeded", 0)
                + getattr(counts, "errored", 0)
                + getattr(counts, "expired", 0)
                + getattr(counts, "canceled", 0)
            )
            completed = getattr(counts, "succeeded", 0)
            failed = getattr(counts, "errored", 0)

        # An "ended" batch with zero successes and some failures is effectively failed
        if batch_status == BatchStatusEnum.COMPLETED and completed == 0 and failed > 0:
            batch_status = BatchStatusEnum.FAILED

        return BatchStatus(
            batch_id=batch_id,
            status=batch_status,
            total=total,
            completed=completed,
            failed=failed,
        )

    def retrieve_batch_results(self, batch_id: str) -> dict[str, str]:
        """Récupère les résultats d'un batch Anthropic terminé."""
        client = self._get_client()
        results = {}

        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if hasattr(result, "result") and result.result:
                msg = result.result
                if hasattr(msg, "message") and msg.message:
                    content = ""
                    for block in msg.message.content:
                        if hasattr(block, "text"):
                            content += block.text
                    results[custom_id] = content
                elif hasattr(msg, "type") and msg.type == "error":
                    error = getattr(msg, "error", {})
                    logger.warning(f"Batch result error for {custom_id}: {error}")
                    results[custom_id] = ""
                else:
                    results[custom_id] = ""
            else:
                results[custom_id] = ""

        logger.info(f"Batch {batch_id} : {len(results)} résultats récupérés")
        return results

    def run_batch_with_polling(
        self,
        requests: list[BatchRequest],
        poll_interval: int = 30,
        timeout: int = 3600,
        fallback_to_realtime: bool = True,
    ) -> dict[str, str]:
        """Cycle complet : soumettre, polling, récupération.

        Args:
            requests: Liste de BatchRequest.
            poll_interval: Intervalle entre les vérifications (secondes).
            timeout: Timeout total (secondes).
            fallback_to_realtime: Si True, repasse en temps réel si le batch échoue.

        Returns:
            Dict {custom_id: contenu_généré}.
        """
        try:
            batch_id = self.submit_batch(requests)
        except Exception as e:
            if fallback_to_realtime:
                logger.warning(f"Soumission batch Anthropic échouée, fallback temps réel : {e}")
                return self._fallback_realtime(requests)
            raise

        start = time.time()
        while time.time() - start < timeout:
            try:
                status = self.poll_batch(batch_id)
                logger.info(
                    f"Batch Anthropic {batch_id}: {status.status.value} "
                    f"({status.completed}/{status.total})"
                )

                if status.status == BatchStatusEnum.COMPLETED:
                    return self.retrieve_batch_results(batch_id)
                elif status.status in (BatchStatusEnum.FAILED, BatchStatusEnum.EXPIRED, BatchStatusEnum.CANCELLED):
                    error_msg = f"Batch Anthropic {batch_id} échoué : {status.status.value}"
                    if fallback_to_realtime:
                        logger.warning(f"{error_msg}, fallback temps réel")
                        return self._fallback_realtime(requests)
                    raise BatchError(error_msg)
            except BatchError:
                raise
            except Exception as e:
                logger.warning(f"Erreur polling batch Anthropic : {e}")

            time.sleep(poll_interval)

        error_msg = f"Timeout batch Anthropic {batch_id} après {timeout}s"
        if fallback_to_realtime:
            logger.warning(f"{error_msg}, fallback temps réel")
            return self._fallback_realtime(requests)
        raise BatchError(error_msg)

    def _fallback_realtime(self, requests: list[BatchRequest]) -> dict[str, str]:
        """Exécute les requêtes en temps réel (fallback si batch échoue)."""
        results = {}
        for req in requests:
            try:
                response = self.generate(
                    prompt=req.prompt,
                    system_prompt=req.system_prompt or None,
                    model=req.model or None,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                results[req.custom_id] = response.content
            except Exception as e:
                logger.error(f"Erreur fallback temps réel pour {req.custom_id}: {e}")
                results[req.custom_id] = ""
        return results
