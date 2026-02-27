"""Orchestrateur multi-agents pour la génération documentaire.

Phase 7 : coordonne les 5 agents spécialisés (Architecte, Rédacteur,
           Vérificateur, Évaluateur, Correcteur) en parallèle via asyncio,
           en respectant le graphe de dépendances (DAG) entre sections.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.core.agent_framework import (
    AgentConfig,
    AgentResult,
    AgentState,
    BaseAgent,
)
from src.core.agents.architect_agent import ArchitectAgent
from src.core.agents.corrector_agent import CorrectorAgent
from src.core.agents.evaluator_agent import EvaluatorAgent
from src.core.agents.verifier_agent import VerifierAgent
from src.core.agents.writer_agent import WriterAgent
from src.core.cost_tracker import CostTracker
from src.core.message_bus import MessageBus
from src.core.tool_dispatcher import ToolDispatcher
from src.providers.base import BaseProvider

logger = logging.getLogger("orchestria")


# ── Résultat de génération ──────────────────────────────────────────────────


@dataclass
class GenerationResult:
    """Résultat complet du pipeline multi-agents."""

    sections: dict = field(default_factory=dict)  # section_id -> contenu final
    architecture: dict = field(default_factory=dict)
    verif_reports: dict = field(default_factory=dict)
    eval_result: dict = field(default_factory=dict)
    corrections_made: dict = field(default_factory=dict)  # section_id -> nb passes
    total_duration_ms: int = 0
    total_cost_usd: float = 0.0
    token_breakdown: dict = field(default_factory=dict)
    agent_timeline: list = field(default_factory=list)
    alerts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sections": self.sections,
            "architecture": self.architecture,
            "verif_reports": self.verif_reports,
            "eval_result": self.eval_result,
            "corrections_made": self.corrections_made,
            "total_duration_ms": self.total_duration_ms,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "token_breakdown": self.token_breakdown,
            "agent_timeline": self.agent_timeline,
            "alerts": self.alerts,
        }


# ── DAG Utilities ───────────────────────────────────────────────────────────


def build_dag(dependances: dict) -> dict:
    """Construit et valide le graphe de dépendances.

    Args:
        dependances: dict {section_id: [ids_requis_avant]}

    Returns:
        dict validé (identique à l'entrée si pas de cycle)

    Raises:
        ValueError: si un cycle est détecté
    """
    # Détection de cycles via DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {node: WHITE for node in dependances}

    def dfs(node: str, path: list) -> None:
        colors[node] = GRAY
        for dep in dependances.get(node, []):
            if dep not in colors:
                continue
            if colors[dep] == GRAY:
                cycle = path[path.index(dep):] + [dep]
                raise ValueError(
                    f"Plan invalide : dépendances cycliques détectées entre "
                    f"[{' ↔ '.join(cycle)}]"
                )
            if colors[dep] == WHITE:
                dfs(dep, path + [dep])
        colors[node] = BLACK

    for node in dependances:
        if colors[node] == WHITE:
            dfs(node, [node])

    return dict(dependances)


def get_ready_sections(
    dag: dict,
    completed: set,
    in_progress: set,
) -> list[str]:
    """Retourne les sections dont toutes les dépendances sont complètes."""
    ready = []
    for section_id, deps in dag.items():
        if section_id in completed or section_id in in_progress:
            continue
        if all(d in completed for d in deps):
            ready.append(section_id)
    return ready


# ── Orchestrateur ───────────────────────────────────────────────────────────


class MultiAgentOrchestrator:
    """Orchestrateur du pipeline multi-agents pour la génération documentaire."""

    def __init__(
        self,
        project_state,
        agent_config: AgentConfig,
        providers: dict[str, BaseProvider],
        progress_callback: Optional[Callable] = None,
    ):
        self.state = project_state
        self.config = agent_config
        self.providers = providers
        self.progress_callback = progress_callback

        self.bus = MessageBus()
        self.agents: dict[str, BaseAgent] = {}
        self._dag: dict = {}
        self._sem = asyncio.Semaphore(agent_config.max_parallel_writers)
        self._result = GenerationResult()
        self._cost_tracker = CostTracker()
        self._done = False
        self._start_time = 0.0

        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Instancie les 5 agents avec leur configuration."""
        agent_defs = {
            "architecte": ArchitectAgent,
            "redacteur": WriterAgent,
            "verificateur": VerifierAgent,
            "evaluateur": EvaluatorAgent,
            "correcteur": CorrectorAgent,
        }

        for agent_name, agent_cls in agent_defs.items():
            cfg = self.config.get_agent_config(agent_name)
            provider_name = cfg.get("provider", "openai")
            provider = self.providers.get(provider_name)

            if not provider:
                # Fallback sur le premier provider disponible
                provider = next(iter(self.providers.values()), None)
                if provider:
                    logger.warning(
                        f"Provider '{provider_name}' non disponible pour {agent_name}, "
                        f"fallback sur '{provider.name}'"
                    )

            if not provider:
                logger.error(f"Aucun provider disponible pour l'agent {agent_name}")
                continue

            kwargs = {
                "name": agent_name,
                "provider": provider,
                "model": cfg.get("model", provider.get_default_model()),
                "max_retries": 2,
                "timeout_s": cfg.get("timeout_s", 120),
                "temperature": cfg.get("temperature", 0.7),
                "max_tokens": cfg.get("max_tokens", 4096),
            }

            if agent_name == "correcteur":
                corpus_text = self._get_corpus_text()
                tool_dispatcher = ToolDispatcher(corpus_text, self.bus)
                kwargs["tool_dispatcher"] = tool_dispatcher
                kwargs["max_tool_calls"] = cfg.get("max_tool_calls", 10)

            agent = agent_cls(**kwargs)
            agent.set_bus(self.bus)
            self.agents[agent_name] = agent

    def _get_corpus_text(self) -> str:
        """Extrait le texte du corpus depuis ProjectState."""
        if self.state.corpus and hasattr(self.state.corpus, "full_text"):
            return self.state.corpus.full_text
        if self.state.corpus and hasattr(self.state.corpus, "chunks"):
            chunks = self.state.corpus.chunks
            if chunks:
                parts = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        parts.append(chunk.get("text", chunk.get("content", "")))
                    elif hasattr(chunk, "text"):
                        parts.append(chunk.text)
                    elif hasattr(chunk, "content"):
                        parts.append(chunk.content)
                return "\n\n".join(parts)
        return ""

    async def run(self) -> GenerationResult:
        """Point d'entrée principal. Exécute le pipeline complet."""
        self._start_time = time.time()
        self._done = False

        try:
            # Étape 1 : Architecture
            architecture = await self._run_architect()
            self._result.architecture = architecture

            # Étape 2 : Génération parallèle
            sections = await self._run_generation_phase(architecture)
            self._result.sections = sections

            # Étape 3 : Vérification parallèle
            verif_reports = await self._run_verification_phase(sections, architecture)
            self._result.verif_reports = verif_reports

            # Étape 4 : Évaluation
            eval_result = await self._run_evaluation(sections, verif_reports)
            self._result.eval_result = eval_result

            # Étape 5 : Correction (conditionnelle)
            sections_to_fix = eval_result.get("sections_a_corriger", [])
            if sections_to_fix and eval_result.get("recommandation") != "exporter":
                corrected = await self._run_correction_phase(
                    sections_to_fix, verif_reports, eval_result, architecture
                )
                self._result.sections.update(corrected)

            # Collecter les alertes
            self._result.alerts = [
                {"section_id": a.payload.get("section_id", ""),
                 "reason": a.payload.get("reason", "")}
                for a in self.bus.get_alerts()
            ]

        except Exception as e:
            logger.error(f"Erreur pipeline multi-agents: {e}")
            self._result.eval_result["error"] = str(e)

        self._result.total_duration_ms = int((time.time() - self._start_time) * 1000)
        self._done = True
        return self._result

    async def _run_architect(self) -> dict:
        """Lance l'Architecte et récupère le plan enrichi + le DAG."""
        agent = self.agents.get("architecte")
        if not agent:
            return self._default_architecture()

        corpus_text = self._get_corpus_text()

        task = {
            "plan": self.state.plan,
            "corpus_text": corpus_text,
            "objective": self.state.plan.objective if self.state.plan else "",
            "description": "Analyse du plan et des dépendances",
        }

        self._log_timeline("architecte", "start", "Analyse du plan")
        result = await agent.run(task)
        self._log_timeline("architecte", "end", f"success={result.success}")
        self._track_cost(result, "architecte")

        if result.success and result.structured_data:
            architecture = result.structured_data
            dependances = architecture.get("dependances", {})
            try:
                self._dag = build_dag(dependances)
            except ValueError as e:
                logger.error(str(e))
                self._dag = {sid: [] for sid in dependances}
            return architecture

        return self._default_architecture()

    def _default_architecture(self) -> dict:
        """Architecture par défaut si l'Architecte échoue."""
        sections = []
        dependances = {}
        if self.state.plan:
            for s in self.state.plan.sections:
                sections.append({
                    "id": s.id,
                    "title": s.title,
                    "longueur_cible": 500,
                    "ton": "analytique",
                    "type": "fond",
                })
                dependances[s.id] = []
        self._dag = dependances
        return {
            "sections": sections,
            "dependances": dependances,
            "zones_risque": [],
            "system_prompt_global": "",
        }

    async def _run_generation_phase(self, architecture: dict) -> dict[str, str]:
        """Lance les Rédacteurs en respectant le DAG de dépendances."""
        sections_data = {s["id"]: s for s in architecture.get("sections", [])}
        system_prompt_global = architecture.get("system_prompt_global", "")
        corpus_text = self._get_corpus_text()

        completed: set[str] = set()
        in_progress: set[str] = set()
        generated: dict[str, str] = {}
        section_summaries: dict[str, str] = {}

        while len(completed) < len(self._dag):
            ready = get_ready_sections(self._dag, completed, in_progress)

            if not ready and not in_progress:
                # Toutes les sections non traitées ont des dépendances non résolues
                remaining = set(self._dag.keys()) - completed
                logger.warning(f"Sections bloquées : {remaining}")
                break

            if not ready:
                # Attendre qu'une section en cours se termine
                await asyncio.sleep(0.1)
                continue

            # Lancer les sections prêtes en parallèle (avec sémaphore)
            tasks = []
            for sid in ready:
                in_progress.add(sid)
                section_info = sections_data.get(sid, {"id": sid, "title": sid})

                # Collecter les résumés des sections prérequises
                prereqs = {}
                for dep_id in self._dag.get(sid, []):
                    if dep_id in section_summaries:
                        prereqs[dep_id] = section_summaries[dep_id]
                    elif dep_id in generated:
                        prereqs[dep_id] = generated[dep_id][:500]

                task_data = {
                    "section_id": sid,
                    "section_title": section_info.get("title", sid),
                    "section_type": section_info.get("type", "fond"),
                    "section_ton": section_info.get("ton", "analytique"),
                    "longueur_cible": section_info.get("longueur_cible", 500),
                    "corpus_text": corpus_text,
                    "system_prompt_global": system_prompt_global,
                    "prerequisite_sections": prereqs,
                    "description": f"Rédaction {sid}",
                }

                tasks.append(self._execute_writer(sid, task_data))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sid_result in results:
                if isinstance(sid_result, Exception):
                    logger.error(f"Erreur rédaction: {sid_result}")
                    continue

                sid, result = sid_result
                in_progress.discard(sid)

                if result.success and result.content:
                    generated[sid] = result.content
                    # Stocker dans le bus pour accès par les autres agents
                    await self.bus.store_section(sid, result.content)
                    # Générer un résumé court
                    section_summaries[sid] = result.content[:500]
                else:
                    logger.warning(f"Rédaction échouée pour {sid}: {result.error}")
                    generated[sid] = f"{{{{GENERATION_FAILED}}}}\n{result.error or ''}"
                    await self.bus.store_section(sid, generated[sid])

                completed.add(sid)
                self._track_cost(result, "redacteur")
                self._log_timeline(
                    "redacteur", "section_done",
                    f"{sid} success={result.success}"
                )

                if self.progress_callback:
                    self.progress_callback(
                        len(completed), len(self._dag), "generation"
                    )

        return generated

    async def _execute_writer(
        self, section_id: str, task: dict
    ) -> tuple[str, AgentResult]:
        """Exécute un Rédacteur en respectant le sémaphore."""
        async with self._sem:
            agent = self.agents.get("redacteur")
            if not agent:
                return section_id, AgentResult(
                    agent_name="redacteur",
                    section_id=section_id,
                    success=False,
                    error="Agent rédacteur non disponible",
                )
            self._log_timeline("redacteur", "start", f"Section {section_id}")
            result = await agent.run(task)
            result.section_id = section_id
            return section_id, result

    async def _run_verification_phase(
        self, sections: dict, architecture: dict
    ) -> dict:
        """Lance les Vérificateurs en parallèle."""
        agent = self.agents.get("verificateur")
        if not agent:
            return {}

        corpus_text = self._get_corpus_text()
        verif_reports = {}

        # Construire les résumés inter-sections
        summaries = "\n".join(
            f"[{sid}]: {content[:300]}" for sid, content in sections.items()
        )

        tasks = []
        for sid, content in sections.items():
            section_info = next(
                (s for s in architecture.get("sections", []) if s.get("id") == sid),
                {"id": sid, "title": sid},
            )
            task_data = {
                "section_id": sid,
                "section_title": section_info.get("title", sid),
                "section_content": content,
                "corpus_text": corpus_text,
                "other_sections_summaries": summaries,
                "description": f"Vérification {sid}",
            }
            tasks.append(self._execute_verifier(sid, task_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in results:
            if isinstance(item, Exception):
                logger.error(f"Erreur vérification: {item}")
                continue
            sid, result = item
            if result.success and result.structured_data:
                verif_reports[sid] = result.structured_data
            self._track_cost(result, "verificateur")
            self._log_timeline(
                "verificateur", "section_done",
                f"{sid} success={result.success}"
            )

        return verif_reports

    async def _execute_verifier(
        self, section_id: str, task: dict
    ) -> tuple[str, AgentResult]:
        """Exécute un Vérificateur avec sémaphore."""
        sem = asyncio.Semaphore(self.config.max_parallel_verifiers)
        async with sem:
            agent = self.agents.get("verificateur")
            result = await agent.run(task)
            result.section_id = section_id
            return section_id, result

    async def _run_evaluation(
        self, sections: dict, verif_reports: dict
    ) -> dict:
        """Lance l'Évaluateur sur le document assemblé."""
        agent = self.agents.get("evaluateur")
        if not agent:
            return {"score_global": 3.5, "recommandation": "exporter"}

        task = {
            "sections": sections,
            "verif_reports": verif_reports,
            "quality_threshold": self.config.quality_threshold,
            "section_correction_threshold": self.config.section_correction_threshold,
            "description": "Évaluation globale",
        }

        self._log_timeline("evaluateur", "start", "Évaluation globale")
        result = await agent.run(task)
        self._track_cost(result, "evaluateur")
        self._log_timeline("evaluateur", "end", f"success={result.success}")

        if result.success and result.structured_data:
            return result.structured_data

        return {"score_global": 3.5, "recommandation": "exporter"}

    async def _run_correction_phase(
        self,
        sections_to_fix: list[str],
        verif_reports: dict,
        eval_result: dict,
        architecture: dict,
    ) -> dict[str, str]:
        """Lance le Correcteur sur les sections défaillantes."""
        agent = self.agents.get("correcteur")
        if not agent:
            return {}

        corrected = {}
        scores_par_section = eval_result.get("scores_par_section", {})

        # Construire les résumés des autres sections
        all_sections = self.bus.get_all_sections()
        other_summary = "\n".join(
            f"[{sid}]: {content[:300]}" for sid, content in all_sections.items()
        )

        for pass_num in range(1, self.config.max_correction_passes + 1):
            still_to_fix = []

            for sid in sections_to_fix:
                section_content = all_sections.get(sid, "")
                if not section_content:
                    continue

                section_info = next(
                    (s for s in architecture.get("sections", [])
                     if s.get("id") == sid),
                    {"id": sid, "title": sid},
                )

                score = scores_par_section.get(sid, 3.0)
                eval_info = f"Score section : {score}/5.0"

                task = {
                    "section_id": sid,
                    "section_title": section_info.get("title", sid),
                    "section_content": section_content,
                    "verif_report": verif_reports.get(sid, {}),
                    "eval_info": eval_info,
                    "other_sections_summary": other_summary,
                    "description": f"Correction {sid} (passe {pass_num})",
                }

                self._log_timeline(
                    "correcteur", "start",
                    f"Section {sid} passe {pass_num}"
                )
                result = await agent.run(task)
                self._track_cost(result, "correcteur")

                if result.success and result.content:
                    corrected[sid] = result.content
                    await self.bus.store_section(sid, result.content)
                    all_sections[sid] = result.content

                    # Enregistrer la correction
                    self._result.corrections_made[sid] = (
                        self._result.corrections_made.get(sid, 0) + 1
                    )
                else:
                    still_to_fix.append(sid)
                    logger.warning(
                        f"Correction échouée pour {sid} passe {pass_num}: "
                        f"{result.error}"
                    )

                self._log_timeline(
                    "correcteur", "end",
                    f"{sid} passe {pass_num} success={result.success}"
                )

            sections_to_fix = still_to_fix
            if not sections_to_fix:
                break

        return corrected

    def _track_cost(self, result: AgentResult, agent_name: str) -> None:
        """Enregistre le coût d'un appel agent."""
        if result.token_input or result.token_output:
            agent = self.agents.get(agent_name)
            if agent:
                cost = self._cost_tracker.calculate_cost(
                    provider=agent.provider.name,
                    model=agent.model,
                    input_tokens=result.token_input,
                    output_tokens=result.token_output,
                )
                result.cost_usd = cost
                self._result.total_cost_usd += cost

                # Token breakdown
                breakdown = self._result.token_breakdown
                if agent_name not in breakdown:
                    breakdown[agent_name] = {
                        "input": 0, "output": 0, "cost_usd": 0.0, "calls": 0,
                    }
                breakdown[agent_name]["input"] += result.token_input
                breakdown[agent_name]["output"] += result.token_output
                breakdown[agent_name]["cost_usd"] += cost
                breakdown[agent_name]["calls"] += 1

    def _log_timeline(self, agent: str, event: str, detail: str = "") -> None:
        """Enregistre un événement dans la timeline."""
        self._result.agent_timeline.append({
            "agent": agent,
            "event": event,
            "detail": detail,
            "timestamp": time.time(),
            "elapsed_ms": int((time.time() - self._start_time) * 1000)
            if self._start_time else 0,
        })

    def estimate_pipeline_cost(
        self,
        corpus_tokens: int,
        section_count: int,
    ) -> dict:
        """Estime le coût du pipeline multi-agents avant lancement."""
        # Estimation par agent
        architect_input = corpus_tokens + 2000
        writer_input_per = corpus_tokens + 1500
        verifier_input_per = corpus_tokens + 1500
        evaluator_input = section_count * 1200 + 1000
        corrector_input_per = 8000  # Accès ciblé, pas le corpus complet

        # Estimation des corrections (30% des sections)
        estimated_corrections = max(1, int(section_count * 0.3))

        total_input = (
            architect_input
            + writer_input_per * section_count
            + verifier_input_per * section_count
            + evaluator_input
            + corrector_input_per * estimated_corrections
        )

        total_output = (
            4000  # Architecte
            + 1500 * section_count  # Rédacteurs
            + 500 * section_count  # Vérificateurs
            + 1000  # Évaluateur
            + 1500 * estimated_corrections  # Correcteurs
        )

        # Coût moyen pondéré (approximation)
        avg_input_cost = 5.0  # $/1M tokens
        avg_output_cost = 15.0  # $/1M tokens
        estimated_cost = (
            (total_input / 1_000_000) * avg_input_cost
            + (total_output / 1_000_000) * avg_output_cost
        )

        return {
            "estimated_usd": round(estimated_cost, 2),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "token_breakdown": {
                "architecte": architect_input,
                "redacteur": writer_input_per * section_count,
                "verificateur": verifier_input_per * section_count,
                "evaluateur": evaluator_input,
                "correcteur": corrector_input_per * estimated_corrections,
            },
            "within_budget": estimated_cost <= self.config.max_cost_usd,
            "budget_usd": self.config.max_cost_usd,
        }

    def get_agent_states(self) -> list[AgentState]:
        """Retourne l'état courant de tous les agents."""
        return [agent.state for agent in self.agents.values()]

    def get_current_metrics(self) -> dict:
        """Retourne les métriques en temps réel."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "elapsed_ms": int(elapsed * 1000),
            "sections_generated": len(self._result.sections),
            "sections_verified": len(self._result.verif_reports),
            "cost_usd": round(self._result.total_cost_usd, 4),
            "total_sections": len(self._dag),
        }

    def is_done(self) -> bool:
        """Retourne True si le pipeline est terminé."""
        return self._done
