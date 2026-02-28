"""Fournisseur OpenAI pour Orchestr'IA.

Phase 2.5 : ajout du support batch via /v1/batches.
"""

import json
import os
import tempfile
import time
import logging
from typing import Optional

from src.providers.base import (
    AIResponse, BaseProvider, BatchRequest, BatchStatus, BatchStatusEnum, BatchError,
)

logger = logging.getLogger("orchestria")


class OpenAIProvider(BaseProvider):
    """Fournisseur OpenAI avec support batch."""

    MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 2.0):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """Génère du contenu via l'API OpenAI avec retry automatique."""
        model = model or self.get_default_model()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = response.choices[0]
                usage = response.usage
                return AIResponse(
                    content=choice.message.content or "",
                    model=model,
                    provider=self.name,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    finish_reason=choice.finish_reason or "",
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(f"Erreur API OpenAI (tentative {attempt + 1}/{self._max_retries + 1}): {e}. Retry dans {delay}s...")
                    time.sleep(delay)

        raise RuntimeError(f"Échec après {self._max_retries + 1} tentatives: {last_error}")

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AIResponse:
        """Génère du contenu JSON structuré via l'API OpenAI (response_format=json_object).

        Utilisé notamment pour l'extraction de métadonnées bibliographiques
        en alternative à GROBID.

        Args:
            prompt: Texte contenant les premières pages du PDF.
            system_prompt: Instructions système.
            model: Modèle à utiliser (par défaut gpt-4o-mini).
            temperature: Température de génération.
            max_tokens: Nombre max de tokens en sortie.

        Returns:
            AIResponse dont le contenu est une chaîne JSON.
        """
        model = model or "gpt-4o-mini"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                choice = response.choices[0]
                usage = response.usage
                return AIResponse(
                    content=choice.message.content or "{}",
                    model=model,
                    provider=self.name,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    finish_reason=choice.finish_reason or "",
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        f"Erreur API OpenAI JSON (tentative {attempt + 1}/"
                        f"{self._max_retries + 1}): {e}. Retry dans {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(f"Échec generate_json après {self._max_retries + 1} tentatives: {last_error}")

    def is_available(self) -> bool:
        """Vérifie si la clé API est configurée."""
        return bool(self._api_key and self._api_key != "sk-your-openai-api-key-here")

    def get_default_model(self) -> str:
        return "gpt-4o"

    def list_models(self) -> list[str]:
        return self.MODELS.copy()

    # ── Phase 2.5 : Support batch OpenAI (/v1/batches) ──

    def supports_batch(self) -> bool:
        return True

    def submit_batch(self, requests: list[BatchRequest]) -> str:
        """Soumet un batch via l'API OpenAI Batch.

        1. Prépare un fichier JSONL avec toutes les requêtes.
        2. Upload le fichier via /v1/files.
        3. Crée le batch via /v1/batches.

        Returns:
            batch_id du batch soumis.
        """
        client = self._get_client()

        # 1. Préparer le fichier JSONL
        jsonl_lines = []
        for req in requests:
            model = req.model or self.get_default_model()
            messages = []
            if req.system_prompt:
                messages.append({"role": "system", "content": req.system_prompt})
            messages.append({"role": "user", "content": req.prompt})

            jsonl_lines.append(json.dumps({
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                },
            }))

        # 2. Upload le fichier JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(jsonl_lines) + "\n")
            jsonl_path = f.name

        try:
            with open(jsonl_path, "rb") as f:
                file_obj = client.files.create(file=f, purpose="batch")
            logger.info(f"Fichier batch uploadé : {file_obj.id} ({len(requests)} requêtes)")
        finally:
            os.unlink(jsonl_path)

        # 3. Créer le batch
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(f"Batch OpenAI soumis : {batch.id}")
        return batch.id

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """Vérifie le statut d'un batch OpenAI."""
        client = self._get_client()
        status = client.batches.retrieve(batch_id)

        status_map = {
            "validating": BatchStatusEnum.SUBMITTED,
            "in_progress": BatchStatusEnum.IN_PROGRESS,
            "finalizing": BatchStatusEnum.IN_PROGRESS,
            "completed": BatchStatusEnum.COMPLETED,
            "failed": BatchStatusEnum.FAILED,
            "expired": BatchStatusEnum.EXPIRED,
            "cancelled": BatchStatusEnum.CANCELLED,
            "cancelling": BatchStatusEnum.CANCELLED,
        }

        batch_status = status_map.get(status.status, BatchStatusEnum.IN_PROGRESS)
        counts = status.request_counts
        total = counts.total if counts else 0
        completed = counts.completed if counts else 0
        failed = counts.failed if counts else 0

        error_msg = ""
        if batch_status == BatchStatusEnum.FAILED:
            errors = status.errors
            if errors and errors.data:
                error_msg = "; ".join(e.message for e in errors.data if e.message)

        return BatchStatus(
            batch_id=batch_id,
            status=batch_status,
            total=total,
            completed=completed,
            failed=failed,
            error_message=error_msg,
        )

    def retrieve_batch_results(self, batch_id: str) -> dict[str, str]:
        """Récupère les résultats d'un batch OpenAI terminé."""
        client = self._get_client()
        status = client.batches.retrieve(batch_id)

        if status.status != "completed":
            raise BatchError(f"Le batch {batch_id} n'est pas terminé (statut: {status.status})")

        if not status.output_file_id:
            raise BatchError(f"Le batch {batch_id} n'a pas de fichier de sortie")

        output_content = client.files.content(status.output_file_id)
        results = {}

        for line in output_content.text.splitlines():
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                custom_id = result["custom_id"]
                response_body = result.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    results[custom_id] = content
                else:
                    error = result.get("error", {})
                    logger.warning(f"Batch result error for {custom_id}: {error}")
                    results[custom_id] = ""
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Erreur parsing résultat batch : {e}")

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
                logger.warning(f"Soumission batch échouée, fallback temps réel : {e}")
                return self._fallback_realtime(requests)
            raise

        start = time.time()
        while time.time() - start < timeout:
            try:
                status = self.poll_batch(batch_id)
                logger.info(
                    f"Batch {batch_id}: {status.status.value} "
                    f"({status.completed}/{status.total})"
                )

                if status.status == BatchStatusEnum.COMPLETED:
                    return self.retrieve_batch_results(batch_id)
                elif status.status in (BatchStatusEnum.FAILED, BatchStatusEnum.EXPIRED, BatchStatusEnum.CANCELLED):
                    error_msg = f"Batch {batch_id} échoué : {status.status.value}"
                    if status.error_message:
                        error_msg += f" — {status.error_message}"
                    if fallback_to_realtime:
                        logger.warning(f"{error_msg}, fallback temps réel")
                        return self._fallback_realtime(requests)
                    raise BatchError(error_msg)
            except BatchError:
                raise
            except Exception as e:
                logger.warning(f"Erreur polling batch : {e}")

            time.sleep(poll_interval)

        # Timeout
        error_msg = f"Timeout batch {batch_id} après {timeout}s"
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
