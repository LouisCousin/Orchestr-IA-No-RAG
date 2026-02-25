"""Orchestration séquentielle du pipeline de génération.

Phase 2 : mode agentique, passes multiples, mode batch, RAG.
Phase 3 : intelligence du pipeline — qualité, factcheck, glossaire,
           personas, citations, feedback loop, HITL journal.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.core.checkpoint_manager import CheckpointManager, CheckpointType
from src.core.corpus_extractor import CorpusExtractor, StructuredCorpus
from src.core.cost_tracker import CostTracker
from src.core.plan_parser import NormalizedPlan, PlanSection
from src.core.prompt_engine import PromptEngine
from src.providers.base import BaseProvider
from src.utils.file_utils import ensure_dir, save_json, load_json
from src.utils.logger import ActivityLog
from src.utils.token_counter import count_tokens

logger = logging.getLogger("orchestria")


@dataclass
class ProjectState:
    """État complet d'un projet."""
    name: str
    user_id: str = "user_default"
    plan: Optional[NormalizedPlan] = None
    corpus: Optional[StructuredCorpus] = None
    generated_sections: dict = field(default_factory=dict)  # section_id → content
    section_summaries: list[str] = field(default_factory=list)
    current_step: str = "init"  # "init", "plan", "corpus", "generation", "review", "export", "done"
    current_section_index: int = 0
    current_pass: int = 1
    config: dict = field(default_factory=dict)
    cost_report: dict = field(default_factory=dict)
    deferred_sections: list[str] = field(default_factory=list)
    rag_coverage: dict = field(default_factory=dict)  # section_id → coverage assessment dict
    # Phase 3 fields
    quality_reports: dict = field(default_factory=dict)   # section_id → quality report dict
    factcheck_reports: dict = field(default_factory=dict)  # section_id → factcheck report dict
    glossary: dict = field(default_factory=dict)           # glossary data
    personas: dict = field(default_factory=dict)           # personas config
    citations: dict = field(default_factory=dict)          # citations resolved
    feedback_history: list = field(default_factory=list)   # feedback loop entries
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "user_id": self.user_id,
            "plan": self.plan.to_dict() if self.plan else None,
            "generated_sections": self.generated_sections,
            "section_summaries": self.section_summaries,
            "current_step": self.current_step,
            "current_section_index": self.current_section_index,
            "current_pass": self.current_pass,
            "config": self.config,
            "cost_report": self.cost_report,
            "deferred_sections": self.deferred_sections,
            "rag_coverage": self.rag_coverage,
            "quality_reports": self.quality_reports,
            "factcheck_reports": self.factcheck_reports,
            "glossary": self.glossary,
            "personas": self.personas,
            "citations": self.citations,
            "feedback_history": self.feedback_history,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectState":
        state = cls(
            name=data.get("name", ""),
            user_id=data.get("user_id", "user_default"),
            generated_sections=data.get("generated_sections", {}),
            section_summaries=data.get("section_summaries", []),
            current_step=data.get("current_step", "init"),
            current_section_index=data.get("current_section_index", 0),
            current_pass=data.get("current_pass", 1),
            config=data.get("config", {}),
            cost_report=data.get("cost_report", {}),
            deferred_sections=data.get("deferred_sections", []),
            rag_coverage=data.get("rag_coverage", {}),
            quality_reports=data.get("quality_reports", {}),
            factcheck_reports=data.get("factcheck_reports", {}),
            glossary=data.get("glossary", {}),
            personas=data.get("personas", {}),
            citations=data.get("citations", {}),
            feedback_history=data.get("feedback_history", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        if data.get("plan"):
            state.plan = NormalizedPlan.from_dict(data["plan"])
        return state


def _normalize_config(config: dict) -> dict:
    """Flatten nested YAML config keys to the flat keys used by the codebase.

    Maps known nested structures (e.g. ``conditional_generation.enabled``)
    to the flat keys the orchestrator and UI expect (e.g.
    ``conditional_generation_enabled``).  Flat keys already present take
    precedence so that values set by the UI are never overwritten.
    """
    mapping = {
        # conditional_generation.*
        ("conditional_generation", "enabled"): "conditional_generation_enabled",
        ("conditional_generation", "sufficient_threshold"): "coverage_sufficient_threshold",
        ("conditional_generation", "insufficient_threshold"): "coverage_insufficient_threshold",
        ("conditional_generation", "min_relevant_blocks"): "coverage_min_blocks",
        # anti_hallucination.*
        ("anti_hallucination", "enabled"): "anti_hallucination_enabled",
        # plan_corpus_linking.* — kept nested (read via .get("plan_corpus_linking", {}))
        # batch.* — kept nested for now
    }

    for (section, key), flat_key in mapping.items():
        if flat_key not in config:
            nested = config.get(section)
            if isinstance(nested, dict) and key in nested:
                config[flat_key] = nested[key]

    return config


class Orchestrator:
    """Orchestre le pipeline de génération séquentielle.

    Phase 2 : supporte le mode agentique, les passes multiples,
    le mode batch, et le RAG via ChromaDB.
    """

    def __init__(
        self,
        provider: BaseProvider,
        project_dir: Path,
        checkpoint_manager: Optional[CheckpointManager] = None,
        cost_tracker: Optional[CostTracker] = None,
        activity_log: Optional[ActivityLog] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider
        self.project_dir = ensure_dir(project_dir)
        self.checkpoint_mgr = checkpoint_manager or CheckpointManager()
        self.cost_tracker = cost_tracker or CostTracker()
        self.activity_log = activity_log or ActivityLog()
        self.config = _normalize_config(config or {})
        self.prompt_engine = PromptEngine(
            persistent_instructions=self.config.get("persistent_instructions", ""),
            anti_hallucination_enabled=self.config.get("anti_hallucination_enabled", True),
        )
        self.rag_engine = None
        self.conditional_generator = None
        self.state: Optional[ProjectState] = None
        self._metadata_store = None  # Phase 2.5 : MetadataStore SQLite
        self._last_plan_context = None  # Phase 2.5 : dernier PlanContext pour affichage UI
        # Phase 3 engines (lazy-initialized)
        self._quality_evaluator = None
        self._factcheck_engine = None
        self._citation_engine = None
        self._glossary_engine = None
        self._persona_engine = None
        self._feedback_engine = None
        self._hitl_journal = None

    def _init_phase3_engines(self) -> None:
        """Initialise les engines Phase 3 si nécessaire."""
        # Quality evaluator
        if self._quality_evaluator is None:
            from src.core.quality_evaluator import QualityEvaluator
            qe_config = self.config.get("quality_evaluation", {})
            self._quality_evaluator = QualityEvaluator(
                provider=self.provider,
                weights=qe_config.get("weights"),
                auto_refine_threshold=qe_config.get("auto_refine_threshold", 3.0),
                evaluation_model=qe_config.get("evaluation_model"),
                enabled=qe_config.get("enabled", True),
            )

        # Factcheck engine
        if self._factcheck_engine is None:
            from src.core.factcheck_engine import FactcheckEngine
            fc_config = self.config.get("factcheck", {})
            self._factcheck_engine = FactcheckEngine(
                provider=self.provider,
                rag_engine=self.rag_engine,
                project_dir=self.project_dir,
                enabled=fc_config.get("enabled", True),
                auto_correct_threshold=fc_config.get("auto_correct_threshold", 80.0),
                max_claims_per_section=fc_config.get("max_claims_per_section", 30),
                factcheck_model=fc_config.get("factcheck_model"),
            )

        # Feedback engine
        if self._feedback_engine is None:
            from src.core.feedback_engine import FeedbackEngine
            fb_config = self.config.get("feedback_loop", {})
            self._feedback_engine = FeedbackEngine(
                provider=self.provider,
                enabled=fb_config.get("enabled", True),
                min_diff_ratio=fb_config.get("min_diff_ratio", 0.15),
                analysis_model=fb_config.get("analysis_model"),
            )

        # HITL journal
        if self._hitl_journal is None:
            from src.core.hitl_journal import HITLJournal
            self._hitl_journal = HITLJournal()

        # B4: Connect HITL journal to checkpoint manager
        if self._hitl_journal and self.checkpoint_mgr._hitl_journal is None:
            self.checkpoint_mgr._hitl_journal = self._hitl_journal
            self.checkpoint_mgr._project_name = self.state.name if self.state else ""

        # B1: Glossary engine
        if self._glossary_engine is None:
            from src.core.glossary_engine import GlossaryEngine
            gl_config = self.config.get("glossary", {})
            self._glossary_engine = GlossaryEngine(
                project_dir=self.project_dir,
                max_terms_per_prompt=gl_config.get("max_terms_per_prompt", 15),
                enabled=gl_config.get("enabled", False),
            )

        # B1: Persona engine
        if self._persona_engine is None:
            from src.core.persona_engine import PersonaEngine
            p_config = self.config.get("personas", {})
            self._persona_engine = PersonaEngine(
                project_dir=self.project_dir,
                enabled=p_config.get("enabled", False),
            )

        # B1: Citation engine
        if self._citation_engine is None:
            from src.core.citation_engine import CitationEngine
            cit_config = self.config.get("citations", {})
            self._citation_engine = CitationEngine(
                metadata_store=self._metadata_store,
                enabled=cit_config.get("enabled", False),
            )

        # B1: Persistent instructions engine
        if not hasattr(self, '_persistent_instructions_engine') or self._persistent_instructions_engine is None:
            from src.core.persistent_instructions import PersistentInstructions
            self._persistent_instructions_engine = PersistentInstructions(
                project_dir=self.project_dir,
            )

        # B2: Re-instantiate PromptEngine with Phase 3 parameters (once only)
        if not getattr(self, '_phase3_prompt_engine_initialized', False):
            cit_config = self.config.get("citations", {})
            self.prompt_engine = PromptEngine(
                persistent_instructions=self.config.get("persistent_instructions", ""),
                anti_hallucination_enabled=self.config.get("anti_hallucination_enabled", True),
                citations_enabled=cit_config.get("enabled", False),
                glossary_engine=self._glossary_engine,
                persona_engine=self._persona_engine,
                persistent_instructions_engine=self._persistent_instructions_engine,
            )
            self._phase3_prompt_engine_initialized = True

    def _init_rag(self) -> None:
        """Initialise le moteur RAG si nécessaire."""
        if self.rag_engine is not None:
            return
        try:
            import chromadb  # noqa: F401 – check availability before creating engine
            from src.core.rag_engine import RAGEngine
            persist_dir = self.project_dir / "chromadb"
            ensure_dir(persist_dir)
            logger.info(f"Initialisation ChromaDB — répertoire de persistance : {persist_dir}")
            self.rag_engine = RAGEngine(
                persist_dir=persist_dir,
                top_k=self.config.get("rag", {}).get("top_k", self.config.get("rag_top_k", 10)),
                relevance_threshold=self.config.get("rag", {}).get("relevance_threshold", self.config.get("rag_relevance_threshold", 0.3)),
                config=self.config,  # Phase 2.5 : transmet toute la config
            )
        except ImportError:
            logger.warning("ChromaDB non disponible, RAG désactivé")

    def _init_conditional_generator(self) -> None:
        """Initialise le générateur conditionnel si nécessaire."""
        if self.conditional_generator is not None:
            return
        from src.core.conditional_generator import ConditionalGenerator
        self.conditional_generator = ConditionalGenerator(
            sufficient_threshold=self.config.get("coverage_sufficient_threshold", 0.5),
            insufficient_threshold=self.config.get("coverage_insufficient_threshold", 0.3),
            min_relevant_blocks=self.config.get("coverage_min_blocks", 3),
            enabled=self.config.get("conditional_generation_enabled", True),
        )

    @property
    def is_agentic(self) -> bool:
        """Vérifie si le mode agentique est activé."""
        return self.config.get("mode", "manual") == "agentic"

    def init_project(self, name: str, plan: NormalizedPlan, corpus: Optional[StructuredCorpus] = None) -> ProjectState:
        """Initialise un nouveau projet."""
        self.state = ProjectState(
            name=name,
            plan=plan,
            corpus=corpus,
            config=self.config,
        )
        self.state.current_step = "plan"
        self.activity_log.info(f"Projet initialisé : {name}")
        self.save_state()
        return self.state

    def index_corpus_rag(self) -> int:
        """Indexe le corpus dans ChromaDB avec chunking sémantique (Phase 2.5).

        Utilise le pipeline : ExtractionResult → semantic_chunker → index_corpus_semantic.
        Fallback sur l'ancien index_corpus() si semantic_chunker échoue.

        Returns:
            Nombre de blocs indexés.
        """
        self._init_rag()
        if not self.rag_engine or not self.state or not self.state.corpus:
            return 0

        # Phase 2.5 : tenter le chunking sémantique
        use_semantic = self.config.get("rag", {}).get("chunking", {}).get("strategy", "semantic") == "semantic"

        if use_semantic:
            try:
                from src.core.semantic_chunker import chunk_document
                from src.core.metadata_store import MetadataStore, DocumentMetadata

                # Initialiser le MetadataStore SQLite
                metadata_store = MetadataStore(str(self.project_dir))

                # Paramètres de chunking depuis la config
                chunking_config = self.config.get("rag", {}).get("chunking", {})
                max_chunk_tokens = chunking_config.get("max_chunk_tokens", 800)
                min_chunk_tokens = chunking_config.get("min_chunk_tokens", 100)
                overlap_sentences = chunking_config.get("overlap_sentences", 2)

                chunks_by_doc = {}
                for ext in self.state.corpus.extractions:
                    doc_id = ext.source_filename

                    # Extraire auteurs et année depuis les métadonnées
                    authors = None
                    year = None
                    if ext.metadata:
                        authors = ext.metadata.get("author") or ext.metadata.get("authors")
                        # Tenter d'extraire l'année depuis les métadonnées ou le nom de fichier
                        date_str = ext.metadata.get("creation_date") or ext.metadata.get("date")
                        if date_str:
                            import re as _re
                            year_match = _re.search(r'(19|20)\d{2}', str(date_str))
                            if year_match:
                                year = int(year_match.group())

                    # Construire la référence APA si possible
                    title = ext.metadata.get("title") if ext.metadata else None
                    apa_reference = None
                    if authors and year:
                        apa_reference = f"{authors} ({year})"
                    elif authors and title:
                        apa_reference = f"{authors} — {title}"

                    # Enregistrer le document dans SQLite
                    doc_meta = DocumentMetadata(
                        doc_id=doc_id,
                        filepath=str(ext.source_filename),
                        filename=ext.source_filename,
                        title=title,
                        authors=json.dumps([authors]) if authors else None,
                        year=year,
                        apa_reference=apa_reference,
                        page_count=ext.page_count,
                        token_count=ext.word_count,
                        char_count=ext.char_count,
                        word_count=ext.word_count,
                        extraction_method=ext.extraction_method,
                        extraction_status=ext.status,
                        hash_binary=ext.hash_binary,
                        hash_textual=ext.hash_text,
                    )
                    metadata_store.add_document(doc_meta)

                    # Chunking sémantique
                    chunks = chunk_document(
                        ext,
                        doc_id=doc_id,
                        max_chunk_tokens=max_chunk_tokens,
                        min_chunk_tokens=min_chunk_tokens,
                        overlap_sentences=overlap_sentences,
                    )
                    if chunks:
                        chunks_by_doc[doc_id] = chunks

                if chunks_by_doc:
                    count = self.rag_engine.index_corpus_semantic(chunks_by_doc, metadata_store)
                    persist_dir = self.project_dir / "chromadb"
                    logger.info(
                        f"Corpus indexé (sémantique) : {count} blocs dans {persist_dir}, "
                        f"métadonnées dans {metadata_store.db_path}"
                    )
                    self.activity_log.info(f"Corpus indexé (sémantique) : {count} blocs")
                    # Stocker metadata_store pour réutilisation (plan_corpus_linker, etc.)
                    self._metadata_store = metadata_store
                    # Phase 3: update citation engine with metadata store
                    if self._citation_engine:
                        self._citation_engine.metadata_store = metadata_store
                    return count

            except Exception as e:
                logger.warning(f"Chunking sémantique échoué, fallback chunking fixe : {e}")

        # Fallback : ancien chunking fixe (Phase 2)
        extractions = []
        for ext in self.state.corpus.extractions:
            extractions.append({
                "text": ext.text,
                "source_file": ext.source_filename,
                "page_count": ext.page_count,
            })

        count = self.rag_engine.index_corpus(extractions)
        self.activity_log.info(f"Corpus indexé dans ChromaDB : {count} blocs")
        return count

    def generate_all_sections(self, pass_number: int = 1) -> dict:
        """Génère toutes les sections du plan séquentiellement.

        Args:
            pass_number: Numéro de la passe (1 = brouillon, 2+ = raffinement).

        Returns:
            Dictionnaire section_id → contenu généré.
        """
        if not self.state or not self.state.plan:
            raise RuntimeError("Projet non initialisé ou plan manquant")

        plan = self.state.plan
        model = self.config.get("model", self.provider.get_default_model())
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 4096)
        target_pages = self.config.get("target_pages")

        self.state.current_step = "generation"
        self.state.current_pass = pass_number
        is_refinement = pass_number > 1

        if is_refinement:
            self.activity_log.info(f"Démarrage de la passe de raffinement #{pass_number}")
            # En raffinement, on regénère toutes les sections
            sections_to_generate = list(plan.sections)
        else:
            self.activity_log.info("Démarrage de la génération séquentielle (brouillon)")
            sections_to_generate = [s for s in plan.sections if s.status != "generated"]

        # Initialize Phase 3 engines before generation (must happen before building system prompt)
        try:
            self._init_phase3_engines()
        except Exception as e:
            logger.warning(f"Initialisation Phase 3 échouée : {e}")

        # Initialiser RAG et génération conditionnelle si corpus disponible
        use_rag = self.state.corpus and self.rag_engine is not None
        if self.state.corpus:
            self._init_rag()
            self._init_conditional_generator()
            if self.rag_engine and self.rag_engine.indexed_count == 0:
                self.index_corpus_rag()
            use_rag = self.rag_engine is not None and self.rag_engine.indexed_count > 0

        # Phase 3: Auto-generate glossary before first generation
        if (
            not is_refinement
            and self._glossary_engine
            and self._glossary_engine.enabled
            and not self._glossary_engine.get_all_terms()
            and self.config.get("glossary", {}).get("auto_generate", True)
        ):
            try:
                terms = self._glossary_engine.generate_from_plan(plan, self.provider)
                added = self._glossary_engine.apply_generated_terms(terms)
                if added:
                    self.state.glossary = {"terms": self._glossary_engine.get_all_terms()}
                    self.activity_log.info(f"Glossaire auto-généré : {added} termes")
            except Exception as e:
                logger.warning(f"Génération automatique du glossaire échouée : {e}")

        for i, section in enumerate(sections_to_generate):
            # Vérifier si la section est reportée (génération conditionnelle)
            if section.id in self.state.deferred_sections and not is_refinement:
                self.activity_log.warning(
                    f"Section {section.id} reportée (corpus insuffisant)",
                    section=section.id,
                )
                continue

            # Stocker l'index absolu dans plan.sections (pas l'index filtré)
            try:
                self.state.current_section_index = plan.sections.index(section)
            except ValueError:
                self.state.current_section_index = i
            section.status = "generating"
            self.activity_log.info(
                f"[Passe {pass_number}] Section {section.id}: {section.title}",
                section=section.id,
            )

            # Récupérer les chunks de corpus (RAG ou fallback simple)
            corpus_chunks = []
            extra_instruction = ""
            if use_rag:
                rag_result = self.rag_engine.search_for_section(
                    section.id, section.title, section.description or ""
                )
                corpus_chunks = [
                    type("Chunk", (), {"text": c["text"], "source_file": c["source_file"]})()
                    for c in rag_result.chunks
                ]

                # Évaluation de la couverture conditionnelle
                if self.conditional_generator and not is_refinement:
                    assessment = self.conditional_generator.assess_coverage(rag_result)
                    self.state.rag_coverage[section.id] = assessment.to_dict()

                    if not assessment.should_generate:
                        section.status = "deferred"
                        if section.id not in self.state.deferred_sections:
                            self.state.deferred_sections.append(section.id)
                        self.activity_log.warning(assessment.message, section=section.id)
                        self.save_state()
                        continue

                    if assessment.extra_prompt_instruction:
                        extra_instruction = assessment.extra_prompt_instruction
                        self.activity_log.warning(assessment.message, section=section.id)
            elif self.state.corpus:
                corpus_chunks = self.state.corpus.get_chunks_for_section(section.title)

            # Build per-section system prompt (with section_id for hierarchical
            # instructions and has_corpus to control anti-hallucination block)
            system_prompt = self.prompt_engine.build_system_prompt(
                has_corpus=bool(corpus_chunks),
                section_id=section.id,
            )

            # Construire le prompt
            if is_refinement:
                prompt = self.prompt_engine.build_refinement_prompt(
                    section=section,
                    plan=plan,
                    draft_content=self.state.generated_sections.get(section.id, ""),
                    corpus_chunks=corpus_chunks,
                    previous_summaries=self.state.section_summaries,
                    target_pages=target_pages,
                    extra_instruction=extra_instruction,
                )
            else:
                prompt = self.prompt_engine.build_section_prompt(
                    section=section,
                    plan=plan,
                    corpus_chunks=corpus_chunks,
                    previous_summaries=self.state.section_summaries,
                    target_pages=target_pages,
                    extra_instruction=extra_instruction,
                )

            # Vérification de la taille du contexte
            prompt_tokens = count_tokens(prompt + system_prompt, model)
            self._check_context_window(prompt_tokens, model)

            # Checkpoint avant génération (si activé et mode manuel)
            if not self.is_agentic and self.checkpoint_mgr.should_pause(CheckpointType.PROMPT_GENERATION):
                checkpoint = self.checkpoint_mgr.create_checkpoint(
                    CheckpointType.PROMPT_GENERATION,
                    content=prompt,
                    section_id=section.id,
                )
                if checkpoint:
                    self.save_state()
                    return self.state.generated_sections

            # Appel API
            try:
                response = self.provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                self.cost_tracker.record(
                    section_id=section.id,
                    model=model,
                    provider=self.provider.name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    task_type="refinement" if is_refinement else "generation",
                )

                # Post-traitement : nettoyage des références [Source N] résiduelles
                from src.utils.reference_cleaner import clean_source_references
                content = clean_source_references(response.content)
                self.state.generated_sections[section.id] = content
                section.status = "generated"
                section.generated_content = content

                task_label = "raffinée" if is_refinement else "générée"
                self.activity_log.success(
                    f"Section {section.id} {task_label} ({response.output_tokens} tokens)",
                    section=section.id,
                )

                # Phase 3 : évaluation qualité et factcheck post-génération
                # (may auto-correct content in agentic mode)
                corrected = self._run_post_generation_evaluation(
                    section, content, plan, corpus_chunks,
                    is_refinement=is_refinement,
                )
                if corrected:
                    content = corrected

                # Générer un résumé pour le contexte (passe 1 uniquement)
                # Done after evaluation so summary reflects final content
                if not is_refinement:
                    summary = self._generate_summary(section, content, model, system_prompt)
                    self.state.section_summaries.append(f"[{section.id}] {section.title}: {summary}")

            except Exception as e:
                section.status = "failed"
                self.activity_log.error(f"Erreur génération section {section.id}: {e}", section=section.id)
                logger.error(f"Erreur génération {section.id}: {e}")
                if self.is_agentic:
                    continue  # En mode agentique, on continue
                else:
                    continue  # En mode manuel aussi, mais pourrait être interrompu

            # Checkpoint après génération (mode manuel uniquement)
            if not self.is_agentic and self.checkpoint_mgr.should_pause(CheckpointType.GENERATION):
                checkpoint = self.checkpoint_mgr.create_checkpoint(
                    CheckpointType.GENERATION,
                    content=content,
                    section_id=section.id,
                    metadata={"tokens": response.output_tokens, "pass": pass_number},
                )
                if checkpoint:
                    self.save_state()
                    return self.state.generated_sections

            self.save_state()

        self.state.current_step = "review"
        self.state.cost_report = self.cost_tracker.report.to_dict()
        self.activity_log.success(f"Passe {pass_number} terminée pour toutes les sections")
        self.save_state()
        return self.state.generated_sections

    def generate_multi_pass(self, num_passes: Optional[int] = None) -> dict:
        """Exécute la génération en passes multiples (brouillon + raffinements).

        Args:
            num_passes: Nombre total de passes. Si None, utilise la config.

        Returns:
            Dictionnaire section_id → contenu final.
        """
        num_passes = num_passes or self.config.get("number_of_passes", 1)

        for pass_num in range(1, num_passes + 1):
            self.activity_log.info(f"=== Passe {pass_num}/{num_passes} ===")
            self.generate_all_sections(pass_number=pass_num)

            # Si un checkpoint a interrompu la génération, on s'arrête
            if self.state and self.state.current_step == "generation":
                return self.state.generated_sections

        return self.state.generated_sections if self.state else {}

    def resume_generation(self, action: str = "approved", modified_content: Optional[str] = None, user_comment: str = "") -> dict:
        """Reprend la génération après un checkpoint.

        Args:
            action: Decision ("approved", "modified", "rejected").
            modified_content: Content modified by user (if action=="modified").
            user_comment: User comment on the checkpoint.
        """
        if self.checkpoint_mgr.has_pending:
            pending = self.checkpoint_mgr.pending_checkpoint
            section_id = pending.get("section_id") if pending else None

            result = self.checkpoint_mgr.resolve_checkpoint(action, modified_content, user_comment)

            # m1: If user modified the content, run feedback analysis
            if action == "modified" and modified_content and section_id and self.state:
                original = self.state.generated_sections.get(section_id, "")
                if original:
                    self._init_phase3_engines()
                    self._run_feedback_analysis(section_id, original, modified_content)
                    self.state.generated_sections[section_id] = modified_content

        return self.generate_all_sections(pass_number=self.state.current_pass if self.state else 1)

    def _run_post_generation_evaluation(
        self,
        section: PlanSection,
        content: str,
        plan: NormalizedPlan,
        corpus_chunks: list,
        is_refinement: bool = False,
    ) -> Optional[str]:
        """Exécute l'évaluation qualité et factcheck après génération (Phase 3).

        Returns:
            Updated content if auto-correction was applied, None otherwise.
        """
        try:
            self._init_phase3_engines()
        except Exception as e:
            logger.warning(f"Initialisation Phase 3 échouée : {e}")
            return None

        # Factcheck
        fc_report = None
        factcheck_score = None
        if self._factcheck_engine and self._factcheck_engine.enabled:
            try:
                fc_report = self._factcheck_engine.check_section(
                    section_id=section.id,
                    content=content,
                    section_title=section.title,
                    section_description=section.description or "",
                )
                self.state.factcheck_reports[section.id] = fc_report.to_dict()
                factcheck_score = fc_report.reliability_score
                self.activity_log.info(
                    f"Factcheck {section.id}: {fc_report.reliability_score:.0f}% "
                    f"({fc_report.total_claims} affirmations)",
                    section=section.id,
                )
            except Exception as e:
                logger.warning(f"Factcheck échoué pour {section.id}: {e}")

        # Quality evaluation
        qr = None
        if self._quality_evaluator and self._quality_evaluator.enabled:
            try:
                qr = self._quality_evaluator.evaluate_section(
                    section=section,
                    content=content,
                    plan=plan,
                    corpus_chunks=corpus_chunks,
                    previous_summaries=self.state.section_summaries,
                    factcheck_score=factcheck_score,
                )
                self.state.quality_reports[section.id] = qr.to_dict()
                self.activity_log.info(
                    f"Qualité {section.id}: {qr.global_score:.2f}/5.0",
                    section=section.id,
                )
            except Exception as e:
                logger.warning(f"Évaluation qualité échouée pour {section.id}: {e}")

        # B3: Auto-correction loop (agentic mode only)
        corrected_content = None
        if self.is_agentic:
            corrected_content = self._auto_correct_if_needed(
                section, content, plan, corpus_chunks, fc_report, qr,
            )
            if corrected_content:
                content = corrected_content

        # B3: Citation resolution after generation (runs on final content,
        # whether original or auto-corrected)
        if self._citation_engine and self._citation_engine.enabled:
            try:
                citations = self._citation_engine.extract_inline_citations(content)
                if citations:
                    resolved = self._citation_engine.resolve_citations(citations)
                    self.state.citations[section.id] = [
                        {"raw": c.raw_text, "doc_id": c.resolved_doc_id}
                        for c in resolved
                    ]
            except Exception as e:
                logger.warning(f"Résolution des citations échouée pour {section.id}: {e}")

        return corrected_content

    def _auto_correct_if_needed(
        self,
        section: PlanSection,
        content: str,
        plan: NormalizedPlan,
        corpus_chunks: list,
        fc_report,
        qr,
    ) -> Optional[str]:
        """B3: Trigger auto-correction if factcheck/quality scores are below thresholds.

        Returns:
            Corrected content string if a correction pass was triggered, None otherwise.
        """
        extra_instructions = []

        # Factcheck auto-correction
        if fc_report and self._factcheck_engine and self._factcheck_engine.should_correct(fc_report):
            correction = self._factcheck_engine.get_correction_instruction(fc_report)
            if correction:
                extra_instructions.append(correction)
                self.activity_log.warning(
                    f"Auto-correction factcheck déclenchée pour {section.id} "
                    f"(score: {fc_report.reliability_score:.0f}%)",
                    section=section.id,
                )

        # Quality auto-refinement
        if qr and self._quality_evaluator and self._quality_evaluator.should_refine(qr):
            recommendations = qr.recommendations
            if recommendations:
                reco_text = "Améliorations requises :\n" + "\n".join(f"- {r}" for r in recommendations)
                extra_instructions.append(reco_text)
                self.activity_log.warning(
                    f"Auto-raffinement qualité déclenché pour {section.id} "
                    f"(score: {qr.global_score:.2f}/5.0)",
                    section=section.id,
                )

        if not extra_instructions:
            return None

        # Trigger a single correction pass
        combined_instruction = "\n\n".join(extra_instructions)
        model = self.config.get("model", self.provider.get_default_model())
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 4096)
        target_pages = self.config.get("target_pages")

        try:
            prompt = self.prompt_engine.build_refinement_prompt(
                section=section,
                plan=plan,
                draft_content=content,
                corpus_chunks=corpus_chunks,
                previous_summaries=self.state.section_summaries,
                target_pages=target_pages,
                extra_instruction=combined_instruction,
            )
            system_prompt = self.prompt_engine.build_system_prompt()

            response = self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self.cost_tracker.record(
                section_id=section.id,
                model=model,
                provider=self.provider.name,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                task_type="auto_correction",
            )

            from src.utils.reference_cleaner import clean_source_references
            corrected = clean_source_references(response.content)
            self.state.generated_sections[section.id] = corrected
            section.generated_content = corrected

            self.activity_log.success(
                f"Auto-correction {section.id} terminée ({response.output_tokens} tokens)",
                section=section.id,
            )
            return corrected

        except Exception as e:
            logger.warning(f"Auto-correction échouée pour {section.id}: {e}")
            return None

    def _run_feedback_analysis(
        self,
        section_id: str,
        original_content: str,
        modified_content: str,
    ) -> None:
        """m1: Trigger feedback engine when user modifies text at a checkpoint."""
        if not self._feedback_engine or not self._feedback_engine.enabled:
            return
        try:
            entry = self._feedback_engine.analyze_modification(
                section_id=section_id,
                original=original_content,
                corrected=modified_content,
            )
            if entry and self.state:
                self.state.feedback_history.append(entry.to_dict())
                self.activity_log.info(
                    f"Feedback analysé pour {section_id}: catégorie={entry.category}",
                    section=section_id,
                )
        except Exception as e:
            logger.warning(f"Analyse feedback échouée pour {section_id}: {e}")

    def _generate_summary(self, section: PlanSection, content: str, model: str, system_prompt: str) -> str:
        """Génère un résumé de section pour le contexte."""
        try:
            summary_prompt = self.prompt_engine.build_summary_prompt(section.title, content)
            response = self.provider.generate(
                prompt=summary_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=0.3,
                max_tokens=200,
            )
            self.cost_tracker.record(
                section_id=section.id,
                model=model,
                provider=self.provider.name,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                task_type="summary",
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Impossible de générer le résumé pour {section.id}: {e}")
            return content[:200] + ("..." if len(content) > 200 else "")

    def _check_context_window(self, token_count: int, model: str) -> None:
        """Vérifie si le prompt ne dépasse pas la fenêtre de contexte."""
        from src.utils.config import load_model_pricing
        pricing = load_model_pricing()
        for provider_models in pricing.values():
            if model in provider_models:
                window = provider_models[model].get("context_window", 0)
                if window and token_count > window * 0.9:
                    self.activity_log.warning(
                        f"Attention : le prompt ({token_count} tokens) approche la limite "
                        f"de contexte du modèle {model} ({window} tokens)"
                    )
                return

    def save_state(self) -> None:
        """Sauvegarde l'état du projet sur disque."""
        if self.state:
            state_path = self.project_dir / "state.json"
            save_json(state_path, self.state.to_dict())

    def load_state(self) -> Optional[ProjectState]:
        """Charge l'état du projet depuis le disque."""
        state_path = self.project_dir / "state.json"
        if state_path.exists():
            data = load_json(state_path)
            self.state = ProjectState.from_dict(data)
            return self.state
        return None

    def generate_plan_from_objective(
        self,
        objective: str,
        target_pages: Optional[float] = None,
        corpus: Optional[StructuredCorpus] = None,
    ) -> NormalizedPlan:
        """Génère un plan à partir d'un objectif en langage naturel.

        Phase 2.5 : utilise plan_corpus_linker pour analyser le corpus
        et produire un plan basé sur le contenu réel des documents.

        Args:
            objective: Description de l'objectif du document.
            target_pages: Nombre de pages cible.
            corpus: Corpus structuré optionnel.
        """
        from src.core.plan_parser import PlanParser

        # Phase 2.5 : tenter la liaison plan-corpus si le RAG est indexé
        plan_corpus_enabled = self.config.get("plan_corpus_linking", {}).get("enabled", True)
        use_plan_corpus = (
            plan_corpus_enabled
            and self._metadata_store is not None
            and self.rag_engine is not None
        )

        if use_plan_corpus:
            try:
                from src.core.plan_corpus_linker import (
                    link_plan_to_corpus,
                    format_plan_context_for_prompt,
                    PLAN_PROMPT_WITH_CORPUS,
                )

                plan_context = link_plan_to_corpus(
                    objective=objective,
                    metadata_store=self._metadata_store,
                    collection=self.rag_engine.collection,
                    config=self.config,
                    provider=self.provider,
                )

                corpus_context = format_plan_context_for_prompt(plan_context)
                prompt = PLAN_PROMPT_WITH_CORPUS.format(
                    objective=objective,
                    corpus_context=corpus_context,
                    target_pages=target_pages or "automatique",
                )

                # Stocker le plan_context pour l'affichage UI
                self._last_plan_context = plan_context

                logger.info(
                    f"Plan-corpus linker : {len(plan_context.themes)} thèmes, "
                    f"{len(plan_context.coverage)} scores de couverture"
                )

            except Exception as e:
                logger.warning(f"Plan-corpus linker échoué, fallback digest : {e}")
                use_plan_corpus = False

        if not use_plan_corpus:
            # Fallback : ancien mécanisme (corpus digest textuel)
            corpus_digest = corpus.get_corpus_digest() if corpus else None
            prompt = self.prompt_engine.build_plan_generation_prompt(
                objective, target_pages, corpus_digest=corpus_digest,
            )

        system_prompt = self.prompt_engine.build_system_prompt()

        response = self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.config.get("model", self.provider.get_default_model()),
            temperature=0.7,
            max_tokens=2000,
        )

        self.cost_tracker.record(
            section_id="plan",
            model=self.config.get("model", self.provider.get_default_model()),
            provider=self.provider.name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            task_type="plan_generation",
        )

        parser = PlanParser()
        plan = parser.parse_text(response.content)
        plan.objective = objective

        if target_pages:
            parser.distribute_page_budget(plan, target_pages)

        return plan
