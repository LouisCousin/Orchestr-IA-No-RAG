"""Orchestrateur multi-agents pour la génération documentaire.

Phase 7 : coordonne les 5 agents spécialisés (Architecte, Rédacteur,
           Vérificateur, Évaluateur, Correcteur) en parallèle via asyncio,
           en respectant le graphe de dépendances (DAG) entre sections.

Phase 8 (v3.0) :
  - Intégration du Context Cache Gemini dans le pipeline multi-agents.
  - HITL : scission du workflow (run_architect_phase / run_generation_phase).
  - Circuit Breaker DAG : annulation en cascade des sections descendantes
    lorsqu'une section mère échoue.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.core.agent_framework import (
    AgentConfig,
    AgentMessage,
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
from src.providers.base import BaseProvider, BatchRequest

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


def get_descendants(dag: dict, failed_node: str) -> set[str]:
    """Retourne tous les descendants (directs et indirects) d'un nœud via BFS.

    Un nœud B est descendant de A si A apparaît dans les dépendances
    (directes ou transitives) de B. Autrement dit, B dépend — directement
    ou indirectement — du résultat de A.

    Args:
        dag: Graphe de dépendances {section_id: [deps_requises]}.
        failed_node: Identifiant du nœud en échec.

    Returns:
        Ensemble des identifiants de toutes les sections descendantes
        (excluant *failed_node* lui-même).
    """
    # Construire le graphe inversé : parent → enfants directs
    children: dict[str, list[str]] = {node: [] for node in dag}
    for node, deps in dag.items():
        for dep in deps:
            if dep in children:
                children[dep].append(node)

    # BFS depuis le nœud en échec
    visited: set[str] = set()
    queue: deque[str] = deque(children.get(failed_node, []))
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(children.get(current, []))

    return visited


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
        self._sem_verif = asyncio.Semaphore(agent_config.max_parallel_verifiers)
        self._result = GenerationResult()
        self._cost_tracker = CostTracker()
        self._done = False
        self._start_time = 0.0
        # Phase 8 : Context Cache Gemini pour le pipeline multi-agents
        self._cache_id: Optional[str] = None

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

    # ── Phase 8 : Context Cache ─────────────────────────────────────────────

    def _init_context_cache(self) -> Optional[str]:
        """Initialise le cache de contexte Gemini si le provider actif le supporte.

        Retourne le cache_id (str) ou None si le caching n'est pas applicable.
        """
        state_config = getattr(self.state, "config", {})
        gemini_cfg = state_config.get("gemini", {})
        if not gemini_cfg.get("caching_enabled", False):
            return None

        google_provider = self.providers.get("google")
        if not google_provider:
            return None

        # Vérifier que le provider Gemini supporte le caching
        if not hasattr(google_provider, "supports_caching") or not google_provider.supports_caching():
            return None

        corpus_text = self._get_corpus_text()
        if not corpus_text:
            return None

        try:
            from src.core.gemini_cache_manager import GeminiCacheManager

            cache_manager = GeminiCacheManager()
            existing_cache = getattr(self.state, "cache_id", None)
            model = gemini_cfg.get("model", "gemini-3.1-pro-preview")
            ttl = gemini_cfg.get("cache_ttl_seconds", 7200)

            # Le system_prompt global sera inclus dans le cache
            system_prompt = (
                "Tu es un assistant spécialisé dans la rédaction documentaire. "
                "Le corpus ci-dessous contient les sources de référence pour "
                "la génération de chaque section."
            )

            cache_id = cache_manager.get_or_create_cache(
                project_id=self.state.name if hasattr(self.state, "name") else "project",
                corpus_xml=corpus_text,
                system_prompt=system_prompt,
                model=model,
                ttl=ttl,
                existing_cache_name=existing_cache,
            )
            logger.info(f"Context Cache Gemini initialisé : {cache_id}")
            # Persister le cache_id dans le state
            if hasattr(self.state, "cache_id"):
                self.state.cache_id = cache_id
            return cache_id

        except Exception as e:
            logger.warning(f"Impossible d'initialiser le Context Cache Gemini : {e}")
            return None

    # ── Pipeline principal ────────────────────────────────────────────────

    async def run(self) -> GenerationResult:
        """Point d'entrée principal. Exécute le pipeline complet.

        Note : pour un workflow HITL (humain dans la boucle), utilisez plutôt
        run_architect_phase() puis run_generation_phase(architecture).
        """
        self._start_time = time.time()
        self._done = False

        try:
            # Étape 0 : Initialiser le cache de contexte Gemini
            self._cache_id = self._init_context_cache()

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
        await self._emit_event({
            "type": "generation_complete",
            "cost": self._result.total_cost_usd,
            "message": "Pipeline multi-agents terminé",
            "data": {
                "duration_ms": self._result.total_duration_ms,
                "sections_count": len(self._result.sections),
            },
        })
        return self._result

    # ── HITL : workflow scindé ────────────────────────────────────────────

    async def run_architect_phase(self) -> dict:
        """Phase 1 du workflow HITL : exécute uniquement l'Architecte.

        Retourne l'architecture (sections + dépendances) et met à jour
        ProjectState avec le statut ``waiting_for_architect_validation``.

        Returns:
            Architecture générée par l'Architecte (sections, dependances, etc.).
        """
        self._start_time = time.time()
        self._done = False

        # Initialiser le cache de contexte
        self._cache_id = self._init_context_cache()

        architecture = await self._run_architect()
        self._result.architecture = architecture

        # Mettre à jour le statut pour signaler l'attente HITL
        if hasattr(self.state, "current_step"):
            self.state.current_step = "waiting_for_architect_validation"
        if hasattr(self.state, "agent_architecture"):
            self.state.agent_architecture = architecture

        return architecture

    async def run_generation_phase(self, architecture: dict) -> GenerationResult:
        """Phase 2 du workflow HITL : exécute la génération après validation.

        Reprend le pipeline à partir de l'architecture validée (potentiellement
        modifiée par l'utilisateur).

        Args:
            architecture: Architecture validée par l'utilisateur.

        Returns:
            GenerationResult complet.
        """
        if not self._start_time:
            self._start_time = time.time()

        try:
            # Re-valider le DAG avec l'architecture potentiellement modifiée
            dependances = architecture.get("dependances", {})
            try:
                self._dag = build_dag(dependances)
            except ValueError as e:
                logger.error(str(e))
                self._dag = {sid: [] for sid in dependances}
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
            await self._emit_event({
                "type": "dag_error",
                "message": f"Erreur pipeline : {e}",
            })

        self._result.total_duration_ms = int((time.time() - self._start_time) * 1000)
        self._done = True
        await self._emit_event({
            "type": "generation_complete",
            "cost": self._result.total_cost_usd,
            "message": "Pipeline multi-agents terminé (HITL)",
            "data": {
                "duration_ms": self._result.total_duration_ms,
                "sections_count": len(self._result.sections),
            },
        })
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
        await self._emit_event({"type": "agent_start", "agent": "architecte", "message": "Analyse du plan"})
        result = await agent.run(task)
        self._log_timeline("architecte", "end", f"success={result.success}")
        self._track_cost(result, "architecte")
        await self._emit_event({
            "type": "agent_success" if result.success else "dag_error",
            "agent": "architecte",
            "cost": result.cost_usd,
            "tokens": result.token_input + result.token_output,
            "message": "Analyse terminée" if result.success else (result.error or "Erreur architecte"),
        })

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
        """Lance les Rédacteurs en respectant le DAG de dépendances.

        Si ``use_batch_api`` est activé dans la config, les sections
        indépendantes au même niveau du DAG sont regroupées dans un lot
        unique soumis à l'API Batch du provider (coût réduit de 50%).
        L'orchestrateur s'arrête alors avec un statut ``waiting_for_batch``.

        Sinon, utilise asyncio.create_task pour un vrai streaming DAG : dès
        qu'une section se termine, les sections dépendantes deviennent
        immédiatement éligibles au lancement, sans attendre la fin du batch
        courant.

        Phase 8 (v3.0) :
          - Injection de cache_id (Gemini) au lieu de corpus_text si disponible.
          - Circuit Breaker : en cas d'échec d'une section, tous les descendants
            dans le DAG sont annulés en cascade (pas d'appel API).
        """
        # ── Mode Batch API (Sprint 2) ──
        if self.config.use_batch_api:
            return await self._run_generation_phase_batch(architecture)

        sections_data = {s["id"]: s for s in architecture.get("sections", [])}
        system_prompt_global = architecture.get("system_prompt_global", "")

        # Phase 8 : passer cache_id si disponible, sinon corpus_text
        corpus_text = self._get_corpus_text() if not self._cache_id else ""

        completed: set[str] = set()
        in_progress: set[str] = set()
        generated: dict[str, str] = {}
        section_summaries: dict[str, str] = {}
        pending_tasks: dict[str, asyncio.Task] = {}  # sid -> Task
        cancelled: set[str] = set()  # sections annulées par le circuit breaker

        while len(completed) < len(self._dag):
            # Lancer toutes les sections prêtes non encore en cours
            ready = get_ready_sections(self._dag, completed, in_progress)

            if not ready and not in_progress:
                remaining = set(self._dag.keys()) - completed
                logger.warning(f"Sections bloquées : {remaining}")
                break

            for sid in ready:
                in_progress.add(sid)
                section_info = sections_data.get(sid, {"id": sid, "title": sid})

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
                    "cache_id": self._cache_id,
                    "system_prompt_global": system_prompt_global,
                    "prerequisite_sections": prereqs,
                    "description": f"Rédaction {sid}",
                }

                pending_tasks[sid] = asyncio.create_task(
                    self._execute_writer(sid, task_data)
                )

            if not pending_tasks:
                await asyncio.sleep(0.1)
                continue

            # Wait for at least one task to complete
            done, _ = await asyncio.wait(
                pending_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                # Find the sid for this completed task
                finished_sid = None
                for sid, t in pending_tasks.items():
                    if t is task:
                        finished_sid = sid
                        break
                if finished_sid is None:
                    continue
                del pending_tasks[finished_sid]

                try:
                    sid_result = task.result()
                except Exception as exc:
                    logger.error(f"Erreur rédaction: {exc}")
                    in_progress.discard(finished_sid)
                    # Circuit Breaker : propager l'échec
                    self._cancel_descendants(
                        finished_sid, completed, generated, cancelled,
                        pending_tasks, in_progress,
                    )
                    completed.add(finished_sid)
                    generated[finished_sid] = (
                        f"{{{{GENERATION_FAILED}}}}\n{exc}"
                    )
                    continue

                sid, result = sid_result
                in_progress.discard(sid)

                if result.success and result.content:
                    generated[sid] = result.content
                    await self.bus.store_section(sid, result.content)
                    section_summaries[sid] = result.content[:500]
                else:
                    logger.warning(f"Rédaction échouée pour {sid}: {result.error}")
                    generated[sid] = f"{{{{GENERATION_FAILED}}}}\n{result.error or ''}"
                    await self.bus.store_section(sid, generated[sid])

                    # ── Circuit Breaker : annulation en cascade ──
                    self._cancel_descendants(
                        sid, completed, generated, cancelled,
                        pending_tasks, in_progress,
                    )

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

                # Phase 3 Sprint 3 : émettre la progression
                progress = len(completed) / max(len(self._dag), 1)
                await self._emit_event({
                    "type": "generation_progress",
                    "progress": progress,
                    "section": sid,
                    "message": f"{len(completed)}/{len(self._dag)} sections",
                    "cost": self._result.total_cost_usd,
                })

        return generated

    def _cancel_descendants(
        self,
        failed_sid: str,
        completed: set[str],
        generated: dict[str, str],
        cancelled: set[str],
        pending_tasks: dict[str, "asyncio.Task"],
        in_progress: set[str],
    ) -> None:
        """Circuit Breaker : annule en cascade tous les descendants d'une section échouée.

        Les descendants sont marqués comme complétés avec le marqueur
        ``{{CANCELLED_DEPENDENCY_FAILED}}`` et un événement est publié
        sur le MessageBus pour chaque section annulée.
        """
        descendants = get_descendants(self._dag, failed_sid)
        if not descendants:
            return

        logger.warning(
            f"Circuit Breaker : section '{failed_sid}' échouée → "
            f"annulation de {len(descendants)} descendant(s) : {descendants}"
        )

        for desc_sid in descendants:
            if desc_sid in cancelled or desc_sid in completed:
                continue

            cancelled.add(desc_sid)
            in_progress.discard(desc_sid)
            completed.add(desc_sid)
            generated[desc_sid] = (
                f"{{{{CANCELLED_DEPENDENCY_FAILED}}}}\n"
                f"Annulé suite à l'échec de la section {failed_sid}"
            )

            # Annuler la tâche asyncio si elle est en cours
            if desc_sid in pending_tasks:
                pending_tasks[desc_sid].cancel()
                del pending_tasks[desc_sid]

            # Publier une alerte sur le MessageBus
            alert = AgentMessage(
                sender="orchestrateur",
                recipient="*",
                type="alert",
                payload={
                    "section_id": desc_sid,
                    "reason": (
                        f"Section '{desc_sid}' annulée suite à l'échec "
                        f"de la section '{failed_sid}'"
                    ),
                },
                section_id=desc_sid,
                priority=1,
            )
            self.bus.store_alert_sync(alert)

            self._log_timeline(
                "circuit_breaker", "cancelled",
                f"{desc_sid} (dépend de {failed_sid})",
            )

            # Émettre l'événement d'annulation via le bus (synchrone, pas d'await ici)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._emit_event({
                    "type": "dag_error",
                    "section": desc_sid,
                    "agent": "circuit_breaker",
                    "message": f"Annulé suite à l'échec de {failed_sid}",
                }))
            except RuntimeError:
                pass

    # ── Batch API Mode (Sprint 2) ─────────────────────────────────────────

    async def _run_generation_phase_batch(self, architecture: dict) -> dict[str, str]:
        """Mode Batch API : soumet les sections par niveaux du DAG.

        Pour chaque « vague » de sections prêtes (sans dépendances en attente),
        construit un lot BatchRequest et le soumet via le provider. Le statut
        du projet passe à ``waiting_for_batch`` et l'orchestrateur s'arrête.

        La reprise se fait via ``resume_from_batch(batch_results)``.

        Returns:
            dict partiel des sections déjà générées (vide au premier lancement,
            complet si reprise batch terminée).
        """
        sections_data = {s["id"]: s for s in architecture.get("sections", [])}
        system_prompt_global = architecture.get("system_prompt_global", "")
        corpus_text = self._get_corpus_text() if not self._cache_id else ""

        completed: set[str] = set()
        generated: dict[str, str] = {}
        section_summaries: dict[str, str] = {}

        # Restaurer l'avancement depuis un batch précédent
        if hasattr(self.state, "batch_generated") and self.state.batch_generated:
            generated.update(self.state.batch_generated)
            completed.update(generated.keys())
            for sid, content in generated.items():
                section_summaries[sid] = content[:500]

        while len(completed) < len(self._dag):
            ready = get_ready_sections(self._dag, completed, set())

            if not ready:
                remaining = set(self._dag.keys()) - completed
                logger.warning(f"Batch mode : sections bloquées : {remaining}")
                break

            # Construire les BatchRequest pour toutes les sections prêtes
            batch_requests: list[BatchRequest] = []
            for sid in ready:
                section_info = sections_data.get(sid, {"id": sid, "title": sid})

                prereqs = {}
                for dep_id in self._dag.get(sid, []):
                    if dep_id in section_summaries:
                        prereqs[dep_id] = section_summaries[dep_id]
                    elif dep_id in generated:
                        prereqs[dep_id] = generated[dep_id][:500]

                prompt = self._build_writer_prompt(
                    sid, section_info, corpus_text, system_prompt_global, prereqs,
                )

                writer_cfg = self.config.get_agent_config("redacteur")
                batch_requests.append(BatchRequest(
                    custom_id=sid,
                    prompt=prompt,
                    system_prompt=system_prompt_global,
                    model=writer_cfg.get("model", "gpt-4.1"),
                    temperature=writer_cfg.get("temperature", 0.7),
                    max_tokens=writer_cfg.get("max_tokens", 4096),
                ))

            # Trouver un provider capable de batch
            batch_provider = self._get_batch_provider()
            if not batch_provider:
                logger.warning("Aucun provider ne supporte le batch, fallback temps réel")
                # Fallback : exécution classique pour cette vague
                for sid in ready:
                    section_info = sections_data.get(sid, {"id": sid, "title": sid})
                    prereqs = {}
                    for dep_id in self._dag.get(sid, []):
                        if dep_id in section_summaries:
                            prereqs[dep_id] = section_summaries[dep_id]
                    task_data = {
                        "section_id": sid,
                        "section_title": section_info.get("title", sid),
                        "section_type": section_info.get("type", "fond"),
                        "section_ton": section_info.get("ton", "analytique"),
                        "longueur_cible": section_info.get("longueur_cible", 500),
                        "corpus_text": corpus_text,
                        "cache_id": self._cache_id,
                        "system_prompt_global": system_prompt_global,
                        "prerequisite_sections": prereqs,
                        "description": f"Rédaction {sid}",
                    }
                    _, result = await self._execute_writer(sid, task_data)
                    if result.success and result.content:
                        generated[sid] = result.content
                        section_summaries[sid] = result.content[:500]
                    completed.add(sid)
                continue

            # Soumettre le lot
            logger.info(
                f"Batch API : soumission de {len(batch_requests)} sections "
                f"({[r.custom_id for r in batch_requests]})"
            )
            batch_id = batch_provider.submit_batch(batch_requests)

            # Mettre en pause : enregistrer l'état pour reprise ultérieure
            if hasattr(self.state, "current_step"):
                self.state.current_step = "waiting_for_batch"
            if hasattr(self.state, "batch_id"):
                self.state.batch_id = batch_id
            else:
                # Stocker dynamiquement si l'attribut n'existe pas encore
                self.state.batch_id = batch_id
            # Persister les sections déjà générées
            self.state.batch_generated = dict(generated)
            self.state.batch_architecture = architecture

            self._log_timeline(
                "batch_api", "submitted",
                f"batch_id={batch_id}, sections={[r.custom_id for r in batch_requests]}",
            )

            # Retourner ce qu'on a pour l'instant — le pipeline s'arrête ici
            return generated

        return generated

    def _build_writer_prompt(
        self,
        sid: str,
        section_info: dict,
        corpus_text: str,
        system_prompt_global: str,
        prereqs: dict,
    ) -> str:
        """Construit le prompt de rédaction pour une section (mode batch)."""
        title = section_info.get("title", sid)
        section_type = section_info.get("type", "fond")
        ton = section_info.get("ton", "analytique")
        longueur = section_info.get("longueur_cible", 500)

        prompt_parts = [
            f"Rédige la section '{title}' (ID: {sid}).",
            f"Type : {section_type}, Ton : {ton}, Longueur cible : ~{longueur} mots.",
        ]
        if corpus_text:
            prompt_parts.append(f"\n--- CORPUS DE RÉFÉRENCE ---\n{corpus_text[:8000]}\n---")
        if prereqs:
            prereq_text = "\n".join(
                f"[{dep_id}]: {summary}" for dep_id, summary in prereqs.items()
            )
            prompt_parts.append(f"\n--- SECTIONS PRÉREQUISES ---\n{prereq_text}\n---")

        return "\n".join(prompt_parts)

    def _get_batch_provider(self) -> Optional[BaseProvider]:
        """Retourne le premier provider disponible supportant le batch."""
        # Privilégier le provider du rédacteur
        writer_cfg = self.config.get_agent_config("redacteur")
        writer_provider_name = writer_cfg.get("provider", "openai")
        writer_provider = self.providers.get(writer_provider_name)
        if writer_provider and hasattr(writer_provider, "supports_batch") and writer_provider.supports_batch():
            return writer_provider
        # Fallback : chercher un provider qui supporte le batch
        for provider in self.providers.values():
            if hasattr(provider, "supports_batch") and provider.supports_batch():
                return provider
        return None

    async def resume_from_batch(
        self,
        batch_results: dict[str, str],
        architecture: Optional[dict] = None,
    ) -> dict[str, str]:
        """Reprend la génération après la complétion d'un batch.

        Intègre les résultats du batch dans ``generated``, puis relance
        ``_run_generation_phase()`` pour débloquer les sections suivantes
        du DAG.

        Args:
            batch_results: Dict {section_id: contenu_généré} du batch terminé.
            architecture: Architecture du pipeline (si None, utilise celle en cache).

        Returns:
            Dict complet de toutes les sections générées.
        """
        if architecture is None:
            architecture = getattr(self.state, "batch_architecture", self._result.architecture)

        # Re-valider le DAG
        dependances = architecture.get("dependances", {})
        try:
            self._dag = build_dag(dependances)
        except ValueError as e:
            logger.error(str(e))
            self._dag = {sid: [] for sid in dependances}

        # Intégrer les résultats du batch
        existing = getattr(self.state, "batch_generated", {}) or {}
        existing.update(batch_results)
        self.state.batch_generated = existing

        # Stocker dans le bus
        for sid, content in batch_results.items():
            await self.bus.store_section(sid, content)
            self._log_timeline("batch_api", "result_received", f"{sid}")

        # Nettoyer l'état batch
        if hasattr(self.state, "current_step"):
            self.state.current_step = "generation"
        if hasattr(self.state, "batch_id"):
            self.state.batch_id = None

        # Relancer la phase de génération (les sections déjà générées seront skippées)
        return await self._run_generation_phase(architecture)

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
            await self._emit_event({
                "type": "agent_start",
                "section": section_id,
                "agent": "writer",
                "message": f"Rédaction de la section {section_id}",
            })
            result = await agent.run(task)
            result.section_id = section_id
            await self._emit_event({
                "type": "agent_success" if result.success else "dag_error",
                "section": section_id,
                "agent": "writer",
                "cost": result.cost_usd,
                "tokens": result.token_input + result.token_output,
                "message": f"Section {section_id} terminée" if result.success else (result.error or "Échec"),
            })
            return section_id, result

    async def _run_verification_phase(
        self, sections: dict, architecture: dict
    ) -> dict:
        """Lance les Vérificateurs en parallèle."""
        agent = self.agents.get("verificateur")
        if not agent:
            return {}

        # Phase 8 : utiliser cache_id si disponible
        corpus_text = self._get_corpus_text() if not self._cache_id else ""
        verif_reports = {}

        # Construire les résumés inter-sections
        summaries = "\n".join(
            f"[{sid}]: {content[:300]}" for sid, content in sections.items()
        )

        # Ne pas vérifier les sections annulées par le Circuit Breaker
        sections_to_verify = {
            sid: content for sid, content in sections.items()
            if "{{CANCELLED_DEPENDENCY_FAILED}}" not in content
            and "{{GENERATION_FAILED}}" not in content
        }

        tasks = []
        for sid, content in sections_to_verify.items():
            section_info = next(
                (s for s in architecture.get("sections", []) if s.get("id") == sid),
                {"id": sid, "title": sid},
            )
            task_data = {
                "section_id": sid,
                "section_title": section_info.get("title", sid),
                "section_content": content,
                "corpus_text": corpus_text,
                "cache_id": self._cache_id,
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
        async with self._sem_verif:
            agent = self.agents.get("verificateur")
            if not agent:
                return section_id, AgentResult(
                    agent_name="verificateur",
                    section_id=section_id,
                    success=False,
                    error="Agent verificateur non disponible",
                )
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

    # ── Phase 3 Sprint 3 : émission d'événements WebSocket ──────────────

    async def _emit_event(self, event: dict) -> None:
        """Émet un événement vers les clients WebSocket connectés."""
        try:
            await self.bus.emit_ws_event(event)
        except Exception as exc:
            logger.debug(f"Émission WS ignorée : {exc}")

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
