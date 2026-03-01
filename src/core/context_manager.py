"""Gestionnaire de contexte unifié pour tous les agents — v4.1.

Remplace RAGEngine. Gère le contexte complet du corpus pour chaque agent,
soit par injection directe (mode STANDARD), soit via un cache Gemini
(mode HIGH_VOLUME_CACHE), soit via un Atlas structuré + Top-K
(mode CORPUS_ATLAS pour les documents extrêmes > 3.2M tokens).
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
    # v4.1 — Mode CORPUS_ATLAS
    atlas_text: Optional[str] = None         # Atlas formaté pour le prompt
    top_k_corpus_text: Optional[str] = None  # Documents Top-K sélectionnés
    memory_text: Optional[str] = None        # DocumentMemory formaté


class ContextManager:
    """Interface unifiée d'accès au corpus pour tous les agents.

    Remplace RAGEngine. Quatre modes :
      - STANDARD : corpus XML en mémoire, injecté dans chaque appel
      - HIGH_VOLUME_CACHE : cache Gemini, référencé par cache_name
      - SEMANTIC_COMPRESSION_REQUIRED : compression puis cache
      - CORPUS_ATLAS : Atlas structuré + sélection Top-K par section
    """

    def __init__(self, strategy: GenerationStrategy, config: dict | None = None):
        self.strategy = strategy
        self._config = config or {}
        self._cache_manager = None
        self._heartbeat = None
        self._corpus_xml: Optional[str] = None
        self._cache_name: Optional[str] = None
        self._created_at: Optional[str] = None
        # v4.1 — Atlas
        self._atlas = None          # CorpusAtlas
        self._corpus = None         # StructuredCorpus (pour Top-K retrieval)
        self._document_memory = None  # DocumentMemory

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
        # Conserver le corpus pour le mode Atlas (Top-K retrieval)
        self._corpus = corpus

        if self.strategy == GenerationStrategy.CORPUS_ATLAS:
            # Mode Atlas : pas de XML complet, pas de cache
            # L'Atlas est construit séparément via set_atlas()
            logger.info(
                "[ContextManager] Mode CORPUS_ATLAS — contexte via Atlas + Top-K"
            )
            return

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
        section_plan: Optional[dict] = None,
    ) -> AgentContextPayload:
        """Retourne le payload de contexte pour un agent.

        Args:
            section_id: ID de la section (pour logging).
            task_type: Type de tâche ("generation", "verification", etc.).
            section_plan: Infos de la section (titre, thèmes, entités) pour Top-K.

        Returns:
            AgentContextPayload avec soit corpus_text, cache_name, ou atlas.
        """
        # v4.1 — Mode Atlas
        if self.strategy == GenerationStrategy.CORPUS_ATLAS and self._atlas:
            return self._prepare_atlas_context(section_id, section_plan)

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

    # ── v4.1 : Atlas support ─────────────────────────────────────────────────

    def set_atlas(self, atlas) -> None:
        """Injecte un CorpusAtlas construit par l'Orchestrateur."""
        self._atlas = atlas

    def set_document_memory(self, memory) -> None:
        """Injecte la DocumentMemory pour la cohérence narrative."""
        self._document_memory = memory

    def get_atlas(self):
        """Retourne le CorpusAtlas si disponible."""
        return self._atlas

    def _prepare_atlas_context(
        self,
        section_id: Optional[str],
        section_plan: Optional[dict],
    ) -> AgentContextPayload:
        """Construit le payload Atlas avec Top-K selection.

        Budget tokens par section :
          - Atlas formaté : ~200K tokens
          - Top-K documents : ~500K tokens
          - DocumentMemory : ~50K tokens
          - Prompt + instructions : ~10K tokens
          Total : ~760K (< 1M fenêtre Gemini)
        """
        from src.core.atlas_builder import AtlasBuilder

        atlas_cfg = self._config.get("corpus_atlas", {})
        top_k = atlas_cfg.get("top_k_documents", 5)
        max_top_k_tokens = atlas_cfg.get("max_top_k_tokens", 500_000)

        # Atlas formaté
        atlas_text = self._atlas.format_for_prompt(max_tokens=200_000)

        # Top-K selection basé sur le plan de la section
        top_k_text = ""
        if section_plan and self._corpus:
            section_title = section_plan.get("title", "")
            section_themes = section_plan.get("themes", [])
            section_entities = section_plan.get("entities", [])

            top_k_sources = AtlasBuilder.select_top_k_documents(
                atlas=self._atlas,
                section_title=section_title,
                section_themes=section_themes,
                section_entities=section_entities,
                k=top_k,
            )

            if top_k_sources:
                top_k_text = self._extract_top_k_text(
                    top_k_sources, max_top_k_tokens
                )

        # DocumentMemory
        memory_text = ""
        if self._document_memory:
            memory_text = self._document_memory.format_for_prompt()

        return AgentContextPayload(
            corpus_text=None,
            cache_name=None,
            strategy="corpus_atlas",
            atlas_text=atlas_text,
            top_k_corpus_text=top_k_text,
            memory_text=memory_text,
        )

    def _extract_top_k_text(
        self,
        source_files: list[str],
        max_tokens: int,
    ) -> str:
        """Extrait le texte des documents Top-K depuis le corpus original."""
        if not self._corpus or not hasattr(self._corpus, "chunks"):
            return ""

        # Regrouper les chunks par source
        chunks_by_source: dict[str, list] = {}
        for chunk in self._corpus.chunks:
            source = getattr(chunk, "source_file", "")
            if source in source_files:
                chunks_by_source.setdefault(source, []).append(chunk)

        parts = []
        token_budget = max_tokens
        for source in source_files:
            if source not in chunks_by_source:
                continue

            doc_parts = [f"═══ DOCUMENT : {source} ═══"]
            for chunk in chunks_by_source[source]:
                text = getattr(chunk, "text", "")
                chunk_tokens = getattr(chunk, "token_estimate", len(text) // 4)
                if token_budget - chunk_tokens < 0:
                    # Tronquer le dernier chunk
                    remaining_chars = token_budget * 4
                    if remaining_chars > 100:
                        doc_parts.append(text[:remaining_chars] + "...")
                    token_budget = 0
                    break
                doc_parts.append(text)
                token_budget -= chunk_tokens

            parts.append("\n".join(doc_parts))
            if token_budget <= 0:
                break

        return "\n\n".join(parts)

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
