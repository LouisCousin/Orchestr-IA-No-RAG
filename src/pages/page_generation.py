"""Page de génération séquentielle du contenu (Phase 2+3 : multi-pass, agentique, RAG, qualité, factcheck)."""

import copy
import logging

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import save_json
from src.utils.logger import ActivityLog
from src.core.orchestrator import Orchestrator
from src.core.corpus_extractor import CorpusExtractor
from src.core.cost_tracker import CostTracker
from src.core.checkpoint_manager import CheckpointManager, CheckpointConfig
from src.utils.providers_registry import PROVIDERS_INFO
from src.utils.reference_cleaner import clean_source_references

logger = logging.getLogger("orchestria")

PROJECTS_DIR = ROOT_DIR / "projects"


def _get_model(config: dict, provider) -> str:
    """Récupère le modèle depuis la config avec warning si absent."""
    model = config.get("model")
    if not model:
        logger.warning("Clé 'model' absente de la config, fallback sur le modèle par défaut du provider")
        model = provider.get_default_model()
    return model


def render():
    st.title("Génération du document")
    st.info(
        "**Étape 4/5** — Lancez la génération du contenu section par section. L'IA "
        "utilise votre corpus et le plan défini pour produire chaque section. "
        "Vous pouvez relire et modifier chaque section après génération."
    )
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

    # Bouton retour en haut
    if st.button("← Retour au plan"):
        st.session_state.current_page = "plan"
        st.rerun()

    _render_generation_config(state, provider)
    st.markdown("---")
    _render_launch_and_progress(state, provider)

    # Sections reportées (génération conditionnelle)
    if state.deferred_sections:
        st.markdown("---")
        _render_deferred_sections(state)

    # Phase 3: Quality and factcheck reports
    if state.generated_sections and (state.quality_reports or state.factcheck_reports):
        st.markdown("---")
        _render_phase3_reports(state)

    # Section relecture (visible si des sections sont générées)
    if state.generated_sections:
        st.markdown("---")
        _render_review(state)


def _render_generation_config(state, provider):
    """Panneau d'ajustement de la configuration avant génération."""
    config = state.config
    provider_name = config.get("default_provider", provider.name)
    info = PROVIDERS_INFO.get(provider_name, {})

    # Liste des modèles depuis le provider ou fallback
    models = provider.list_models() if provider else info.get("models", [])

    # Lecture stricte du modèle depuis la config (ne jamais écraser silencieusement)
    if "model" not in config:
        st.warning("Aucun modèle configuré. Veuillez repasser par la page Configuration pour sélectionner un modèle.")
        current_model = models[0] if models else provider.get_default_model()
    else:
        current_model = config["model"]
        if current_model not in models:
            st.warning(f"Le modèle '{current_model}' n'est pas disponible pour le fournisseur actuel. Sélectionnez-en un autre.")
            current_model = models[0] if models else provider.get_default_model()

    with st.expander("Ajuster la configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox(
                "Modèle",
                models,
                index=models.index(current_model) if current_model in models else 0,
                key="gen_model",
            )
            temperature = st.slider(
                "Température", 0.0, 1.0,
                value=config.get("temperature", 0.7),
                step=0.1,
                help="0 = déterministe, 1 = créatif",
                key="gen_temperature",
            )
        with col2:
            max_tokens = st.number_input(
                "Tokens max par section", 512, 16384,
                value=config.get("max_tokens", 4096),
                step=512,
                key="gen_max_tokens",
            )
            num_passes = st.number_input(
                "Nombre de passes",
                min_value=1, max_value=5,
                value=config.get("number_of_passes", 1),
                help="1 = brouillon seul, 2+ = brouillon + raffinement(s)",
                key="gen_num_passes",
            )

        # Détecter les changements et sauvegarder
        changed = (
            model != config.get("model")
            or temperature != config.get("temperature", 0.7)
            or max_tokens != config.get("max_tokens", 4096)
            or num_passes != config.get("number_of_passes", 1)
        )

        if changed:
            if st.button("Appliquer les modifications", type="primary", key="apply_gen_config"):
                config["model"] = model
                config["temperature"] = temperature
                config["max_tokens"] = max_tokens
                config["number_of_passes"] = num_passes
                state.config = config
                project_id = st.session_state.current_project
                save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
                st.success("Configuration mise à jour.")
                st.rerun()


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
    col3.metric("Modèle", _get_model(config, provider))
    col4.metric("Mode", "Agentique" if mode == "agentic" else "Manuel")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Passes", f"{state.current_pass}/{num_passes}")
    col6.metric("Pages cibles", config.get("target_pages") or "Auto")
    col7.metric("Fournisseur", provider.name.capitalize())
    col8.metric("Reportées", deferred)

    effective_total = total_sections - deferred
    if effective_total > 0:
        st.progress(min(already_generated / effective_total, 1.0), text=f"{already_generated}/{effective_total} sections")

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
        model=_get_model(config, provider),
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
        config=copy.deepcopy(state.config),
    )
    orchestrator.state = state

    # Stocker l'orchestrateur en session (Bug #1 fix)
    st.session_state["orchestrator"] = orchestrator

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

    # Initialize Phase 3 engines (glossary, persona, persistent instructions,
    # citations, etc.) BEFORE the generation loop so that the PromptEngine
    # already has Phase 3 parameters when building prompts for the first section.
    orchestrator._ensure_phase3_engine()

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
                # Pass RAG result dicts directly — the prompt engine's
                # _get_chunk_attr already handles both dicts and objects.
                corpus_chunks = rag_result.chunks

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

            system_prompt = orchestrator.prompt_engine.build_system_prompt(
                has_corpus=bool(corpus_chunks),
                section_id=section.id,
            )

            try:
                response = provider.generate(
                    prompt=prompt, system_prompt=system_prompt,
                    model=_get_model(state.config, provider),
                    temperature=state.config.get("temperature", 0.7),
                    max_tokens=state.config.get("max_tokens", 4096),
                )

                tracker.record(
                    section_id=section.id,
                    model=_get_model(state.config, provider),
                    provider=provider.name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    task_type="refinement" if is_refinement else "generation",
                )

                # Post-traitement : nettoyage des références [Source N] résiduelles
                cleaned_content = clean_source_references(response.content)
                state.generated_sections[section.id] = cleaned_content
                section.status = "generated"
                section.generated_content = cleaned_content

                if not is_refinement:
                    # Générer un résumé par l'IA pour un meilleur contexte inter-sections
                    try:
                        summary_prompt = orchestrator.prompt_engine.build_summary_prompt(
                            section.title, response.content
                        )
                        summary_response = provider.generate(
                            prompt=summary_prompt,
                            system_prompt=orchestrator.prompt_engine.build_system_prompt(has_corpus=False),
                            model=_get_model(state.config, provider),
                            temperature=0.3,
                            max_tokens=200,
                        )
                        summary = summary_response.content.strip()
                        tracker.record(
                            section_id=section.id,
                            model=_get_model(state.config, provider),
                            provider=provider.name,
                            input_tokens=summary_response.input_tokens,
                            output_tokens=summary_response.output_tokens,
                            task_type="summary",
                        )
                    except Exception:
                        summary = response.content[:200] + "..."
                    state.section_summaries.append(f"[{section.id}] {section.title}: {summary}")

                task_label = "raffinée" if is_refinement else "générée"
                activity_log.success(
                    f"Section {section.id} {task_label} ({response.output_tokens} tokens)",
                    section=section.id,
                )

                # Phase 3: run post-generation evaluation (quality + factcheck)
                try:
                    orchestrator._init_phase3_engines()
                    corrected = orchestrator._run_post_generation_evaluation(
                        section, cleaned_content, plan, corpus_chunks,
                        is_refinement=is_refinement,
                    )
                    if corrected:
                        state.generated_sections[section.id] = corrected
                        section.generated_content = corrected
                except Exception as eval_err:
                    logger.warning(f"Phase 3 evaluation failed for {section.id}: {eval_err}")

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


def _render_phase3_reports(state):
    """Phase 3: Display quality and factcheck reports at the GENERATION_REVIEW checkpoint."""
    st.subheader("Rapports de qualité et vérification factuelle")

    plan = state.plan
    for section in plan.sections:
        sid = section.id
        qr = state.quality_reports.get(sid)
        fcr = state.factcheck_reports.get(sid)
        if not qr and not fcr:
            continue

        with st.expander(f"{sid} {section.title} — Rapports", expanded=False):
            col_q, col_f = st.columns(2)

            # Quality report
            with col_q:
                st.markdown("**Rapport qualité**")
                if qr:
                    score = qr.get("global_score", 0)
                    color = "green" if score >= 4.0 else ("orange" if score >= 3.0 else "red")
                    st.markdown(f"Score global : **:{color}[{score:.2f}/5.0]**")

                    for criterion in qr.get("criteria", []):
                        c_score = criterion.get("score", 0)
                        c_name = criterion.get("name", "")
                        c_just = criterion.get("justification", "")
                        bar_val = c_score / 5.0
                        st.progress(bar_val, text=f"{c_name}: {c_score:.1f}/5")
                        if c_just:
                            st.caption(c_just)

                    recommendations = qr.get("recommendations", [])
                    if recommendations:
                        st.markdown("**Recommandations :**")
                        for rec in recommendations:
                            st.markdown(f"- {rec}")

                    ns_count = qr.get("needs_source_count", 0)
                    if ns_count > 0:
                        st.warning(f"{ns_count} marqueur(s) {{{{NEEDS_SOURCE}}}} détecté(s)")
                else:
                    st.info("Non disponible")

            # Factcheck report
            with col_f:
                st.markdown("**Rapport factcheck**")
                if fcr:
                    reliability = fcr.get("reliability_score", 0)
                    total_claims = fcr.get("total_claims", 0)
                    fc_color = "green" if reliability >= 80 else ("orange" if reliability >= 60 else "red")
                    st.markdown(f"Fiabilité : **:{fc_color}[{reliability:.0f}%]** ({total_claims} affirmations)")

                    status_counts = fcr.get("status_counts", {})
                    for status, count in status_counts.items():
                        if count > 0:
                            icon = {"CORROBORÉE": "green", "PLAUSIBLE": "blue", "NON FONDÉE": "orange", "CONTREDITE": "red"}.get(status, "gray")
                            st.markdown(f":{icon}[{status}] : {count}")

                    # Details with color highlighting
                    details = fcr.get("details", [])
                    problematic = [d for d in details if d.get("status") in ("NON FONDÉE", "CONTREDITE")]
                    if problematic:
                        st.markdown("**Affirmations problématiques :**")
                        for claim in problematic:
                            status = claim.get("status", "")
                            text = claim.get("text", "")
                            just = claim.get("justification", "")
                            claim_color = "orange" if status == "NON FONDÉE" else "red"
                            st.markdown(f"- :{claim_color}[[{status}]] {text}")
                            if just:
                                st.caption(f"  → {just}")
                else:
                    st.info("Non disponible")


def _render_review(state):
    """Relecture des sections générées avec feedback loop (Phase 3)."""
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
                # Phase 3: Show diff summary
                original_words = len(content.split())
                edited_words = len(edited.split())
                diff_words = abs(edited_words - original_words)
                st.caption(f"Modification détectée : {original_words} → {edited_words} mots (delta: {diff_words})")

                col_save, col_reject = st.columns(2)
                with col_save:
                    if st.button("Accepter les modifications", key=f"save_{section.id}", type="primary"):
                        # Phase 3: Run feedback analysis
                        orchestrator = st.session_state.get("orchestrator")
                        if orchestrator and hasattr(orchestrator, '_run_feedback_analysis'):
                            try:
                                orchestrator._init_phase3_engines()
                                orchestrator._run_feedback_analysis(section.id, content, edited)
                            except Exception as e:
                                logger.warning(f"Feedback analysis failed for {section.id}: {e}")

                        state.generated_sections[section.id] = edited
                        section.generated_content = edited
                        project_id = st.session_state.current_project
                        save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
                        st.success(f"Section {section.id} mise à jour.")
                        st.rerun()

                with col_reject:
                    if st.button("Rejeter (restaurer)", key=f"reject_{section.id}"):
                        st.rerun()

    # Phase 3: Feedback loop statistics
    if state.feedback_history:
        with st.expander("Feedback loop — Historique"):
            for entry in state.feedback_history[-10:]:
                cat = entry.get("category", "")
                suggestion = entry.get("suggestion", "")
                sid = entry.get("section_id", "")
                st.markdown(f"- **{sid}** [{cat}] : {suggestion}")

    st.markdown("---")
    if st.button("Passer à l'export", type="primary", use_container_width=True):
        state.current_step = "export"
        project_id = st.session_state.current_project
        save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
        st.session_state.current_page = "export"
        st.rerun()
