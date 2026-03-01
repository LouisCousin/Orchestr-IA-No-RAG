"""Gestionnaire de contexte unifié pour tous les agents — v4.0.

Remplace RAGEngine. Gère le contexte complet du corpus pour chaque agent,
soit par injection directe (mode STANDARD), soit via un cache Gemini
(mode HIGH_VOLUME_CACHE).
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.strategy_selector import GenerationStrategy

logger = logging.getLogger("orchestria")


@dataclass
class AgentContextPayload:
    """Objet standard passé à chaque agent."""
    corpus_text: Optional[str] = None  # Mode STANDARD
    cache_name: Optional[str] = None   # Mode HIGH_VOLUME_CACHE
    strategy: str = "standard"


class ContextManager:
    """Interface unifiée d'accès au corpus pour tous les agents.

    Remplace RAGEngine. Trois modes :
      - STANDARD : corpus XML en mémoire, injecté dans chaque appel
      - HIGH_VOLUME_CACHE : cache Gemini, référencé par cache_name
      - SEMANTIC_COMPRESSION_REQUIRED : compression puis cache
    """

    def __init__(self, strategy: GenerationStrategy, config: dict | None = None):
        self.strategy = strategy
        self._config = config or {}
        self._cache_manager = None
        self._heartbeat = None
        self._corpus_xml: Optional[str] = None
        self._cache_name: Optional[str] = None
        self._created_at: Optional[str] = None

    async def prepare(
        self,
        corpus,
        system_instruction: str,
        plan=None,
    ) -> None:
        """Prépare le contexte selon la stratégie sélectionnée.

        Args:
            corpus: StructuredCorpus contenant les documents.
            system_instruction: System prompt global stable.
            plan: NormalizedPlan (optionnel, pour enrichir le cache).
        """
        # Formater le corpus en XML structuré
        self._corpus_xml = self._format_corpus_xml(corpus)

        if self.strategy == GenerationStrategy.STANDARD:
            logger.info(
                f"[ContextManager] Mode STANDARD — corpus en mémoire "
                f"({len(self._corpus_xml)} chars)"
            )
            return

        if self.strategy in (
            GenerationStrategy.HIGH_VOLUME_CACHE,
            GenerationStrategy.SEMANTIC_COMPRESSION_REQUIRED,
        ):
            await self._create_cache(system_instruction, plan)

    async def _create_cache(
        self,
        system_instruction: str,
        plan=None,
    ) -> None:
        """Crée un cache Gemini avec le corpus XML."""
        from src.core.cache_heartbeat import CacheHeartbeat
        from src.core.gemini_cache_manager import GeminiCacheManager

        self._cache_manager = GeminiCacheManager()

        # Enrichir le XML avec le plan si disponible
        cache_content = self._corpus_xml
        if plan and hasattr(plan, "to_dict"):
            plan_xml = self._format_plan_xml(plan)
            cache_content = f"{self._corpus_xml}\n\n{plan_xml}"

        gemini_cfg = self._config.get("gemini", {})
        model = gemini_cfg.get("model", "gemini-3.1-pro-preview")
        ttl = gemini_cfg.get("cache_ttl_seconds", 7200)

        self._cache_name = self._cache_manager.create_corpus_cache(
            project_id="orchestria",
            corpus_xml=cache_content,
            system_prompt=system_instruction,
            model=model,
            ttl=ttl,
        )
        self._created_at = datetime.now(timezone.utc).isoformat()

        # Démarrer le heartbeat
        heartbeat_enabled = gemini_cfg.get("heartbeat_enabled", True)
        if heartbeat_enabled:
            self._heartbeat = CacheHeartbeat(
                self._cache_manager, self._cache_name, self._config
            )
            self._heartbeat.start()

        logger.info(
            f"[ContextManager] Mode HIGH_VOLUME_CACHE — cache créé : "
            f"{self._cache_name}"
        )

    async def upgrade_to_cache(
        self,
        plan=None,
        system_instruction: str = "",
    ) -> None:
        """Upgrade de STANDARD vers HIGH_VOLUME_CACHE après planification.

        Appelé quand l'output estimé dépasse le seuil de 50K tokens.
        """
        if self._cache_name:
            return  # Déjà en mode cache

        self.strategy = GenerationStrategy.HIGH_VOLUME_CACHE
        await self._create_cache(system_instruction, plan)

    def get_context_for_agent(
        self,
        section_id: Optional[str] = None,
        task_type: str = "generation",
    ) -> AgentContextPayload:
        """Retourne le payload de contexte pour un agent.

        Args:
            section_id: ID de la section (pour logging).
            task_type: Type de tâche ("generation", "verification", etc.).

        Returns:
            AgentContextPayload avec soit corpus_text soit cache_name.
        """
        if self._cache_name:
            return AgentContextPayload(
                corpus_text=None,
                cache_name=self._cache_name,
                strategy="high_volume_cache",
            )
        return AgentContextPayload(
            corpus_text=self._corpus_xml,
            cache_name=None,
            strategy="standard",
        )

    async def cleanup(self) -> None:
        """Supprime le cache Gemini si actif et arrête le heartbeat."""
        if self._heartbeat:
            self._heartbeat.stop()
            self._heartbeat = None

        if self._cache_manager and self._cache_name:
            try:
                self._cache_manager.delete_cache(self._cache_name)
                logger.info(
                    f"[ContextManager] Cache supprimé : {self._cache_name}"
                )
            except Exception as e:
                logger.warning(
                    f"[ContextManager] Échec suppression cache : {e}"
                )
            self._cache_name = None

    def get_cache_id(self) -> Optional[str]:
        """Retourne le cache_name Gemini si actif, None sinon."""
        return self._cache_name

    def get_cache_lifecycle(self) -> dict:
        """Retourne les informations de cycle de vie du cache."""
        return {
            "cache_name": self._cache_name,
            "created_at": self._created_at,
            "renewal_count": (
                self._heartbeat.renewal_count if self._heartbeat else 0
            ),
            "heartbeat_active": (
                self._heartbeat.is_running if self._heartbeat else False
            ),
        }

    @staticmethod
    def _format_corpus_xml(corpus) -> str:
        """Formate le corpus en XML structuré pour injection ou cache.

        Accepte un StructuredCorpus ou un objet similaire avec des chunks.
        """
        root = ET.Element("corpus")

        if corpus is None:
            return ET.tostring(root, encoding="unicode", xml_declaration=False)

        # Regrouper les chunks par document source
        docs_by_source: dict[str, list] = {}
        if hasattr(corpus, "chunks"):
            for chunk in corpus.chunks:
                source = getattr(chunk, "source_file", "unknown")
                docs_by_source.setdefault(source, []).append(chunk)

        for doc_idx, (source, chunks) in enumerate(docs_by_source.items()):
            doc_elem = ET.SubElement(root, "document")
            doc_elem.set("id", f"doc_{doc_idx:03d}")
            doc_elem.set("source", source)

            for chunk in chunks:
                chunk_elem = ET.SubElement(doc_elem, "passage")
                chunk_elem.set(
                    "page", str(getattr(chunk, "page_number", ""))
                )
                chunk_elem.set(
                    "section", getattr(chunk, "section_title", "")
                )
                text = getattr(chunk, "text", "")
                if not text and hasattr(chunk, "content"):
                    text = chunk.content
                chunk_elem.text = text

        return ET.tostring(root, encoding="unicode", xml_declaration=False)

    @staticmethod
    def _format_plan_xml(plan) -> str:
        """Formate le plan en XML pour enrichir le cache."""
        root = ET.Element("plan")
        if hasattr(plan, "title"):
            root.set("title", plan.title or "")
        if hasattr(plan, "objective"):
            root.set("objective", plan.objective or "")

        if hasattr(plan, "sections"):
            for section in plan.sections:
                sec_elem = ET.SubElement(root, "section")
                sec_elem.set("id", section.id)
                sec_elem.set("title", section.title)
                sec_elem.set("level", str(section.level))
                if section.description:
                    sec_elem.text = section.description

        return ET.tostring(root, encoding="unicode", xml_declaration=False)
