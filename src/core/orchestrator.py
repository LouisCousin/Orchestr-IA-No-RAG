"""Orchestration séquentielle du pipeline de génération.

Phase 2 : mode agentique, passes multiples, mode batch, RAG.
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
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        if data.get("plan"):
            state.plan = NormalizedPlan.from_dict(data["plan"])
        return state


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
        self.config = config or {}
        self.prompt_engine = PromptEngine(
            persistent_instructions=self.config.get("persistent_instructions", "")
        )
        self.rag_engine = None
        self.conditional_generator = None
        self.state: Optional[ProjectState] = None

    def _init_rag(self) -> None:
        """Initialise le moteur RAG si nécessaire."""
        if self.rag_engine is not None:
            return
        try:
            import chromadb  # noqa: F401 – check availability before creating engine
            from src.core.rag_engine import RAGEngine
            persist_dir = self.project_dir / "chromadb"
            self.rag_engine = RAGEngine(
                persist_dir=persist_dir,
                top_k=self.config.get("rag_top_k", 7),
                relevance_threshold=self.config.get("rag_relevance_threshold", 0.3),
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
        """Indexe le corpus dans ChromaDB pour le RAG.

        Returns:
            Nombre de blocs indexés.
        """
        self._init_rag()
        if not self.rag_engine or not self.state or not self.state.corpus:
            return 0

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

        system_prompt = self.prompt_engine.build_system_prompt()

        # Initialiser RAG et génération conditionnelle si corpus disponible
        use_rag = self.state.corpus and self.rag_engine is not None
        if self.state.corpus:
            self._init_rag()
            self._init_conditional_generator()
            if self.rag_engine and self.rag_engine.indexed_count == 0:
                self.index_corpus_rag()
            use_rag = self.rag_engine is not None and self.rag_engine.indexed_count > 0

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

                content = response.content
                self.state.generated_sections[section.id] = content
                section.status = "generated"
                section.generated_content = content

                task_label = "raffinée" if is_refinement else "générée"
                self.activity_log.success(
                    f"Section {section.id} {task_label} ({response.output_tokens} tokens)",
                    section=section.id,
                )

                # Générer un résumé pour le contexte (passe 1 uniquement)
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

    def resume_generation(self) -> dict:
        """Reprend la génération après un checkpoint."""
        if self.checkpoint_mgr.has_pending:
            self.checkpoint_mgr.resolve_checkpoint("approved")
        return self.generate_all_sections(pass_number=self.state.current_pass if self.state else 1)

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

        Args:
            objective: Description de l'objectif du document.
            target_pages: Nombre de pages cible.
            corpus: Corpus structuré optionnel. Si fourni, des extraits
                représentatifs de chaque document seront injectés dans le
                prompt pour guider la structure du plan.
        """
        from src.core.plan_parser import PlanParser

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
