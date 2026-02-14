"""Orchestration séquentielle du pipeline de génération."""

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
    config: dict = field(default_factory=dict)
    cost_report: dict = field(default_factory=dict)
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
            "config": self.config,
            "cost_report": self.cost_report,
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
            config=data.get("config", {}),
            cost_report=data.get("cost_report", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        if data.get("plan"):
            state.plan = NormalizedPlan.from_dict(data["plan"])
        return state


class Orchestrator:
    """Orchestre le pipeline de génération séquentielle."""

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
        self.state: Optional[ProjectState] = None

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

    def generate_all_sections(self) -> dict:
        """Génère toutes les sections du plan séquentiellement.

        Retourne un dictionnaire section_id → contenu généré.
        """
        if not self.state or not self.state.plan:
            raise RuntimeError("Projet non initialisé ou plan manquant")

        plan = self.state.plan
        model = self.config.get("model", self.provider.get_default_model())
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 4096)
        target_pages = self.config.get("target_pages")

        self.state.current_step = "generation"
        self.activity_log.info("Démarrage de la génération séquentielle")

        system_prompt = self.prompt_engine.build_system_prompt()

        # Filtrer les sections de niveau feuille (ou toutes si pas de hiérarchie complexe)
        sections_to_generate = [s for s in plan.sections if s.status != "generated"]

        for i, section in enumerate(sections_to_generate):
            self.state.current_section_index = i
            section.status = "generating"
            self.activity_log.info(f"Génération section {section.id}: {section.title}", section=section.id)

            # Récupérer les chunks de corpus pertinents
            corpus_chunks = []
            if self.state.corpus:
                corpus_chunks = self.state.corpus.get_chunks_for_section(section.title)

            # Construire le prompt
            prompt = self.prompt_engine.build_section_prompt(
                section=section,
                plan=plan,
                corpus_chunks=corpus_chunks,
                previous_summaries=self.state.section_summaries,
                target_pages=target_pages,
            )

            # Vérification de la taille du contexte
            prompt_tokens = count_tokens(prompt + system_prompt, model)
            self._check_context_window(prompt_tokens, model)

            # Checkpoint avant génération (si activé)
            if self.checkpoint_mgr.should_pause(CheckpointType.PROMPT_GENERATION):
                checkpoint = self.checkpoint_mgr.create_checkpoint(
                    CheckpointType.PROMPT_GENERATION,
                    content=prompt,
                    section_id=section.id,
                )
                if checkpoint:
                    self.save_state()
                    return self.state.generated_sections  # Pause pour intervention

            # Appel API
            try:
                response = self.provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Enregistrer le coût
                self.cost_tracker.record(
                    section_id=section.id,
                    model=model,
                    provider=self.provider.name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    task_type="generation",
                )

                content = response.content
                self.state.generated_sections[section.id] = content
                section.status = "generated"
                section.generated_content = content

                self.activity_log.success(
                    f"Section {section.id} générée ({response.output_tokens} tokens)",
                    section=section.id,
                )

                # Générer un résumé pour le contexte des sections suivantes
                summary = self._generate_summary(section, content, model, system_prompt)
                self.state.section_summaries.append(f"[{section.id}] {section.title}: {summary}")

            except Exception as e:
                section.status = "failed"
                self.activity_log.error(f"Erreur génération section {section.id}: {e}", section=section.id)
                logger.error(f"Erreur génération {section.id}: {e}")
                continue

            # Checkpoint après génération (si activé)
            if self.checkpoint_mgr.should_pause(CheckpointType.GENERATION):
                checkpoint = self.checkpoint_mgr.create_checkpoint(
                    CheckpointType.GENERATION,
                    content=content,
                    section_id=section.id,
                    metadata={"tokens": response.output_tokens},
                )
                if checkpoint:
                    self.save_state()
                    return self.state.generated_sections  # Pause pour intervention

            self.save_state()

        self.state.current_step = "review"
        self.state.cost_report = self.cost_tracker.report.to_dict()
        self.activity_log.success("Génération complète de toutes les sections")
        self.save_state()
        return self.state.generated_sections

    def resume_generation(self) -> dict:
        """Reprend la génération après un checkpoint."""
        if self.checkpoint_mgr.has_pending:
            # Résoudre automatiquement le checkpoint en cours
            self.checkpoint_mgr.resolve_checkpoint("approved")
        return self.generate_all_sections()

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
            return content[:200] + "..."

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

    def generate_plan_from_objective(self, objective: str, target_pages: Optional[float] = None) -> NormalizedPlan:
        """Génère un plan à partir d'un objectif en langage naturel."""
        from src.core.plan_parser import PlanParser

        prompt = self.prompt_engine.build_plan_generation_prompt(objective, target_pages)
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
