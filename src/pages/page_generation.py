"""Page de génération séquentielle du contenu (Phase 2 : multi-pass, agentique, RAG)."""

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import save_json
from src.utils.logger import ActivityLog
from src.core.orchestrator import Orchestrator
from src.core.corpus_extractor import CorpusExtractor
from src.core.cost_tracker import CostTracker
from src.core.checkpoint_manager import CheckpointManager, CheckpointConfig


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Génération du document")
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif.")
        return

    state = st.session_state.project_state
    provider = st.session_state.get("provider")

    if not state.plan:
        st.warning("Aucun plan défini. Rendez-vous sur la page Plan.")
        return

    if not provider or not provider.is_available():
        st.warning("Fournisseur IA non configuré. Rendez-vous sur la page Configuration.")
        return

    _render_launch_and_progress(state, provider)

    # Sections reportées (génération conditionnelle)
    if state.deferred_sections:
        st.markdown("---")
        _render_deferred_sections(state)

    # Section relecture (visible si des sections sont générées)
    if state.generated_sections:
        st.markdown("---")
        _render_review(state)


def _render_launch_and_progress(state, provider):
    """Interface de lancement et suivi de la génération."""
    plan = state.plan
    config = state.config

    total_sections = len(plan.sections)
    already_generated = len(state.generated_sections)
    deferred = len(state.deferred_sections)
    num_passes = config.get("number_of_passes", 1)
    mode = config.get("mode", "manual")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sections", total_sections)
    col2.metric("Générées", already_generated)
    col3.metric("Modèle", config.get("model", provider.get_default_model()))
    col4.metric("Mode", "Agentique" if mode == "agentic" else "Manuel")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Passes", f"{state.current_pass}/{num_passes}")
    col6.metric("Pages cibles", config.get("target_pages") or "Auto")
    col7.metric("Fournisseur", provider.name.capitalize())
    col8.metric("Reportées", deferred)

    effective_total = total_sections - deferred
    if effective_total > 0:
        st.progress(already_generated / effective_total, text=f"{already_generated}/{effective_total} sections")

    # Estimation des coûts
    tracker = st.session_state.get("cost_tracker") or CostTracker()

    project_id = st.session_state.current_project
    corpus_dir = PROJECTS_DIR / project_id / "corpus"
    avg_corpus_tokens = 2000

    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        if not state.corpus:
            extractor = CorpusExtractor()
            corpus = extractor.extract_corpus(corpus_dir)
            state.corpus = corpus
        if state.corpus and state.corpus.total_chunks > 0:
            avg_corpus_tokens = state.corpus.total_tokens // max(1, total_sections)

    estimate = tracker.estimate_project_cost(
        section_count=total_sections,
        avg_corpus_tokens=avg_corpus_tokens,
        provider=config.get("default_provider", provider.name),
        model=config.get("model", provider.get_default_model()),
        num_passes=num_passes,
    )

    if "error" not in estimate:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Coût estimé", f"${estimate['estimated_cost_usd']:.4f}")
        col2.metric("Tokens input estimés", f"{estimate['estimated_input_tokens']:,}")
        col3.metric("Tokens output estimés", f"{estimate['estimated_output_tokens']:,}")

    cost_report = state.cost_report
    if cost_report and cost_report.get("total_cost_usd", 0) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tokens input réels", f"{cost_report.get('total_input_tokens', 0):,}")
        col2.metric("Tokens output réels", f"{cost_report.get('total_output_tokens', 0):,}")
        col3.metric("Coût réel", f"${cost_report.get('total_cost_usd', 0):.4f}")

    # Couverture RAG par section
    if state.rag_coverage:
        with st.expander("Couverture RAG par section"):
            for sid, cov in state.rag_coverage.items():
                level = cov.get("level", "unknown")
                icon = {"sufficient": "OK", "low": "!", "insufficient": "X"}.get(level, "?")
                score = cov.get("avg_score", 0)
                blocks = cov.get("num_relevant_blocks", 0)
                st.text(f"[{icon}] {sid} — score: {score:.2f}, blocs: {blocks} ({level})")

    # Actions
    st.markdown("---")

    if already_generated > 0 and already_generated < effective_total:
        st.info(f"{already_generated}/{effective_total} sections déjà générées. La génération reprendra.")

    if already_generated >= effective_total and already_generated > 0:
        st.success("Toutes les sections ont été générées !")
        if st.button("Passer à l'export", type="primary", use_container_width=True):
            st.session_state.current_page = "export"
            st.rerun()
    else:
        label = "Lancer la génération"
        if num_passes > 1:
            label = f"Lancer la génération ({num_passes} passes)"

        if st.button(label, type="primary", use_container_width=True):
            _run_generation(state, provider, tracker)

    # Journal d'activité
    with st.expander("Journal d'activité"):
        logs = st.session_state.activity_log.get_recent(30)
        if logs:
            for log in reversed(logs):
                icon = {"info": "i", "warning": "!", "error": "x", "success": "v"}.get(log["level"], "i")
                st.text(f"[{icon}] [{log['timestamp'][11:19]}] {log['message']}")
        else:
            st.info("Aucune activité enregistrée.")


def _run_generation(state, provider, tracker):
    """Exécute la génération séquentielle (avec support multi-pass et RAG)."""
    project_id = st.session_state.current_project
    project_dir = PROJECTS_DIR / project_id

    cp_config = CheckpointConfig.from_dict(state.config.get("checkpoints", {}))
    checkpoint_mgr = CheckpointManager(config=cp_config)
    activity_log = st.session_state.activity_log

    orchestrator = Orchestrator(
        provider=provider,
        project_dir=project_dir,
        checkpoint_manager=checkpoint_mgr,
        cost_tracker=tracker,
        activity_log=activity_log,
        config=state.config,
    )
    orchestrator.state = state

    # Extraction du corpus si nécessaire
    if not state.corpus:
        corpus_dir = project_dir / "corpus"
        if corpus_dir.exists():
            extractor = CorpusExtractor()
            state.corpus = extractor.extract_corpus(corpus_dir)

    # Initialiser le RAG et indexer le corpus
    if state.corpus:
        orchestrator._init_rag()
        if orchestrator.rag_engine:
            orchestrator._init_conditional_generator()
            if orchestrator.rag_engine.indexed_count == 0:
                with st.spinner("Indexation du corpus dans ChromaDB..."):
                    count = orchestrator.index_corpus_rag()
                    st.info(f"Corpus indexé : {count} blocs dans ChromaDB")

    num_passes = state.config.get("number_of_passes", 1)

    progress_bar = st.progress(0, text="Démarrage de la génération...")
    status_area = st.empty()
    log_area = st.empty()

    plan = state.plan

    for pass_num in range(1, num_passes + 1):
        state.current_pass = pass_num
        is_refinement = pass_num > 1

        if is_refinement:
            status_area.info(f"**Passe de raffinement {pass_num}/{num_passes}**")
            sections_to_process = list(plan.sections)
        else:
            sections_to_process = [
                s for s in plan.sections
                if s.status != "generated" and s.id not in state.deferred_sections
            ]

        total = len(sections_to_process)

        for i, section in enumerate(sections_to_process):
            progress_bar.progress(i / max(total, 1), text=f"[Passe {pass_num}] {section.id}: {section.title}")
            task = "Raffinement" if is_refinement else "Génération"
            status_area.info(f"{task} de **{section.id} {section.title}**...")

            corpus_chunks = []
            extra_instruction = ""

            if orchestrator.rag_engine and orchestrator.rag_engine.indexed_count > 0:
                rag_result = orchestrator.rag_engine.search_for_section(
                    section.id, section.title, section.description or ""
                )
                corpus_chunks = [
                    type("Chunk", (), {"text": c["text"], "source_file": c["source_file"]})()
                    for c in rag_result.chunks
                ]

                if orchestrator.conditional_generator and not is_refinement:
                    assessment = orchestrator.conditional_generator.assess_coverage(rag_result)
                    state.rag_coverage[section.id] = assessment.to_dict()

                    if not assessment.should_generate:
                        section.status = "deferred"
                        if section.id not in state.deferred_sections:
                            state.deferred_sections.append(section.id)
                        activity_log.warning(assessment.message, section=section.id)
                        save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
                        continue

                    if assessment.extra_prompt_instruction:
                        extra_instruction = assessment.extra_prompt_instruction
            elif state.corpus:
                corpus_chunks = state.corpus.get_chunks_for_section(section.title)

            if is_refinement:
                prompt = orchestrator.prompt_engine.build_refinement_prompt(
                    section=section, plan=plan,
                    draft_content=state.generated_sections.get(section.id, ""),
                    corpus_chunks=corpus_chunks,
                    previous_summaries=state.section_summaries,
                    target_pages=state.config.get("target_pages"),
                    extra_instruction=extra_instruction,
                )
            else:
                prompt = orchestrator.prompt_engine.build_section_prompt(
                    section=section, plan=plan,
                    corpus_chunks=corpus_chunks,
                    previous_summaries=state.section_summaries,
                    target_pages=state.config.get("target_pages"),
                    extra_instruction=extra_instruction,
                )

            system_prompt = orchestrator.prompt_engine.build_system_prompt()

            try:
                response = provider.generate(
                    prompt=prompt, system_prompt=system_prompt,
                    model=state.config.get("model", provider.get_default_model()),
                    temperature=state.config.get("temperature", 0.7),
                    max_tokens=state.config.get("max_tokens", 4096),
                )

                tracker.record(
                    section_id=section.id,
                    model=state.config.get("model", provider.get_default_model()),
                    provider=provider.name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    task_type="refinement" if is_refinement else "generation",
                )

                state.generated_sections[section.id] = response.content
                section.status = "generated"
                section.generated_content = response.content

                if not is_refinement:
                    summary = response.content[:200] + "..."
                    state.section_summaries.append(f"[{section.id}] {section.title}: {summary}")

                task_label = "raffinée" if is_refinement else "générée"
                activity_log.success(
                    f"Section {section.id} {task_label} ({response.output_tokens} tokens)",
                    section=section.id,
                )

            except Exception as e:
                section.status = "failed"
                activity_log.error(f"Erreur section {section.id}: {e}", section=section.id)
                status_area.error(f"Erreur sur la section {section.id}: {e}")

            state.cost_report = tracker.report.to_dict()
            save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())

            recent_logs = activity_log.get_recent(5)
            log_text = "\n".join(f"[{l['level'].upper()}] {l['message']}" for l in recent_logs)
            log_area.code(log_text, language=None)

        progress_bar.progress(1.0, text=f"Passe {pass_num}/{num_passes} terminée")

    progress_bar.progress(1.0, text="Génération terminée !")
    status_area.success(f"Toutes les passes terminées. Coût total : ${tracker.report.total_cost_usd:.4f}")

    state.current_step = "review"
    save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())


def _render_deferred_sections(state):
    """Affiche les sections reportées par la génération conditionnelle."""
    st.subheader("Sections reportées (corpus insuffisant)")
    st.warning(f"{len(state.deferred_sections)} section(s) reportée(s) faute de couverture dans le corpus.")

    for sid in state.deferred_sections:
        section = state.plan.get_section(sid) if state.plan else None
        title = section.title if section else sid
        cov = state.rag_coverage.get(sid, {})
        score = cov.get("avg_score", 0)
        blocks = cov.get("num_relevant_blocks", 0)
        st.text(f"  [X] {sid} {title} — score: {score:.2f}, blocs: {blocks}")


def _render_review(state):
    """Relecture des sections générées."""
    st.subheader("Relecture des sections")

    plan = state.plan
    for section in plan.sections:
        content = state.generated_sections.get(section.id, "")
        if not content:
            continue

        with st.expander(f"{section.id} {section.title}", expanded=False):
            edited = st.text_area(
                f"Contenu de la section {section.id}",
                value=content,
                height=300,
                key=f"review_{section.id}",
            )

            if edited != content:
                if st.button("Sauvegarder les modifications", key=f"save_{section.id}"):
                    state.generated_sections[section.id] = edited
                    section.generated_content = edited
                    project_id = st.session_state.current_project
                    save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
                    st.success(f"Section {section.id} mise à jour.")

    st.markdown("---")
    if st.button("Passer à l'export", type="primary", use_container_width=True):
        state.current_step = "export"
        project_id = st.session_state.current_project
        save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
        st.session_state.current_page = "export"
        st.rerun()
