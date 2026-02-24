"""Page de configuration — Paramètres du projet et du fournisseur IA (Phase 2)."""

import os
import streamlit as st

from src.core.checkpoint_manager import CheckpointConfig
from src.core.cost_tracker import CostTracker
from src.utils.config import ROOT_DIR
from src.utils.file_utils import save_json
from src.utils.providers_registry import PROVIDERS_INFO


PROJECTS_DIR = ROOT_DIR / "projects"


def _save_state(state):
    """Sauvegarde l'état du projet sur disque."""
    project_id = st.session_state.current_project
    if project_id:
        state_path = PROJECTS_DIR / project_id / "state.json"
        save_json(state_path, state.to_dict())


def _create_provider(provider_name: str, api_key: str):
    """Crée une instance du fournisseur sélectionné."""
    if provider_name == "openai":
        from src.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key)
    elif provider_name == "anthropic":
        from src.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key)
    elif provider_name == "google":
        from src.providers.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=api_key)
    return None


def render():
    st.title("Configuration")
    st.info(
        "**Étape 1/5** — Configurez votre fournisseur IA (clé API), choisissez le "
        "modèle de génération et ajustez les paramètres (température, tokens max). "
        "Ces réglages seront utilisés pour toute la génération du document."
    )
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    _render_api_config()
    st.markdown("---")
    _render_model_config()
    st.markdown("---")
    _render_mode_config()
    st.markdown("---")

    with st.expander("Points de controle (HITL)", expanded=False):
        _render_checkpoint_config()

    with st.expander("Charte graphique", expanded=False):
        _render_styling_config()

    with st.expander("RAG et génération conditionnelle", expanded=False):
        _render_rag_config()

    with st.expander("Export / Import de configuration", expanded=False):
        _render_config_export_import()

    # Navigation inter-étapes
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour à l'accueil", use_container_width=True):
            st.session_state.current_page = "accueil"
            st.rerun()
    with col_next:
        config = st.session_state.project_state.config
        provider = st.session_state.get("provider")
        if provider and provider.is_available():
            if st.button("Passer à l'acquisition du corpus →", type="primary", use_container_width=True):
                _save_state(st.session_state.project_state)
                st.session_state.current_page = "acquisition"
                st.rerun()
        else:
            st.button("Passer à l'acquisition du corpus →", type="primary", use_container_width=True, disabled=True)
            st.caption("Configurez et validez votre fournisseur IA avant de continuer.")


def _render_api_config():
    """Configuration du fournisseur et de la clé API."""
    st.subheader("Fournisseur IA")

    config = st.session_state.project_state.config
    current_provider = config.get("default_provider", "openai")

    provider_names = list(PROVIDERS_INFO.keys())
    provider_labels = [PROVIDERS_INFO[p]["label"] for p in provider_names]

    current_idx = provider_names.index(current_provider) if current_provider in provider_names else 0
    selected_label = st.selectbox("Fournisseur", provider_labels, index=current_idx)
    selected_provider = provider_names[provider_labels.index(selected_label)]

    info = PROVIDERS_INFO[selected_provider]

    # Vérifier la clé dans l'environnement
    api_key_env = os.environ.get(info["env_var"], "")
    has_env_key = bool(api_key_env and api_key_env != info["placeholder"])

    if has_env_key:
        st.success(f"Clé API {info['label']} détectée dans le fichier .env")
        masked = api_key_env[:8] + "..." + api_key_env[-4:] if len(api_key_env) > 12 else "***"
        st.code(masked)
    else:
        st.warning(f"Aucune clé API {info['label']} détectée. Saisissez votre clé ou configurez {info['env_var']} dans .env")

    api_key_input = st.text_input(
        f"Clé API {info['label']} (session uniquement)",
        type="password",
        help="La clé est stockée en session uniquement et n'est pas sauvegardée sur le disque.",
        key=f"api_key_{selected_provider}",
    )

    if st.button("Valider la connexion", type="primary"):
        key = api_key_input or api_key_env
        if not key:
            st.error("Veuillez saisir une clé API.")
            return

        provider = _create_provider(selected_provider, key)
        if provider and provider.is_available():
            st.session_state.provider = provider
            st.session_state.cost_tracker = CostTracker()
            old_provider = config.get("default_provider")
            config["default_provider"] = selected_provider
            # Mettre à jour le modèle si le fournisseur a changé
            if old_provider != selected_provider:
                default_model = PROVIDERS_INFO[selected_provider]["default_model"]
                config["model"] = default_model
            st.session_state.project_state.config = config
            _save_state(st.session_state.project_state)
            st.success(f"Connexion {info['label']} validée !")
        else:
            st.error("Clé API invalide ou vide.")


def _render_model_config():
    """Configuration du modèle IA."""
    st.subheader("Modèle de génération")

    config = st.session_state.project_state.config
    current_provider_name = config.get("default_provider", "openai")
    info = PROVIDERS_INFO.get(current_provider_name, PROVIDERS_INFO["openai"])

    provider = st.session_state.get("provider")
    if provider:
        models = provider.list_models()
    else:
        models = info["models"]

    current_model = config.get("model", info["default_model"])
    if current_model not in models:
        current_model = models[0]
        # Corriger automatiquement le modèle invalide dans la config
        config["model"] = current_model
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)

    model = st.selectbox("Modèle", models, index=models.index(current_model) if current_model in models else 0)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Température", 0.0, 1.0,
            value=config.get("temperature", 0.7),
            step=0.1,
            help="0 = déterministe, 1 = créatif",
        )
    with col2:
        max_tokens = st.number_input(
            "Tokens max par section", 512, 16384,
            value=config.get("max_tokens", 4096),
            step=512,
        )

    num_passes = st.number_input(
        "Nombre de passes de génération",
        min_value=1, max_value=5,
        value=config.get("number_of_passes", 1),
        help="1 = brouillon seul, 2+ = brouillon + raffinement(s)",
    )

    if st.button("Sauvegarder la configuration du modèle"):
        config["model"] = model
        config["temperature"] = temperature
        config["max_tokens"] = max_tokens
        config["number_of_passes"] = num_passes
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Configuration sauvegardée.")


def _render_mode_config():
    """Configuration du mode (manuel / agentique)."""
    st.subheader("Mode d'exécution")

    config = st.session_state.project_state.config
    current_mode = config.get("mode", "manual")

    mode = st.radio(
        "Mode du pipeline",
        ["manual", "agentic"],
        index=0 if current_mode == "manual" else 1,
        format_func=lambda x: "Manuel" if x == "manual" else "Agentique",
        help=(
            "**Manuel** : contrôle étape par étape avec validation à chaque checkpoint.\n"
            "**Agentique** : l'IA orchestre le pipeline de bout en bout, "
            "avec consultation uniquement aux checkpoints activés."
        ),
        horizontal=True,
    )

    if mode == "agentic":
        st.info(
            "En mode agentique, seuls les checkpoints 'Validation du plan' et "
            "'Relecture finale' sont activés par défaut. Les erreurs API sont "
            "gérées automatiquement avec retry."
        )

    if st.button("Appliquer le mode"):
        config["mode"] = mode
        if mode == "agentic":
            config.setdefault("checkpoints", {})
            config["checkpoints"]["after_plan_validation"] = True
            config["checkpoints"]["after_corpus_acquisition"] = False
            config["checkpoints"]["after_extraction"] = False
            config["checkpoints"]["after_prompt_generation"] = False
            config["checkpoints"]["after_generation"] = False
            config["checkpoints"]["final_review"] = True
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success(f"Mode {'agentique' if mode == 'agentic' else 'manuel'} activé.")


def _render_checkpoint_config():
    """Configuration des checkpoints HITL."""
    st.markdown("Activez les étapes où vous souhaitez valider manuellement le résultat.")

    config = st.session_state.project_state.config
    checkpoints = config.get("checkpoints", {})

    cp_plan = st.checkbox("Validation du plan", value=checkpoints.get("after_plan_validation", True))
    cp_corpus = st.checkbox("Après acquisition du corpus", value=checkpoints.get("after_corpus_acquisition", False))
    cp_extraction = st.checkbox("Après extraction", value=checkpoints.get("after_extraction", False))
    cp_prompt = st.checkbox("Après génération des prompts", value=checkpoints.get("after_prompt_generation", False))
    cp_gen = st.checkbox("Après génération de chaque section", value=checkpoints.get("after_generation", False))
    cp_final = st.checkbox("Relecture finale", value=checkpoints.get("final_review", True))

    if st.button("Sauvegarder les checkpoints"):
        config["checkpoints"] = {
            "after_plan_validation": cp_plan,
            "after_corpus_acquisition": cp_corpus,
            "after_extraction": cp_extraction,
            "after_prompt_generation": cp_prompt,
            "after_generation": cp_gen,
            "final_review": cp_final,
        }
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Checkpoints sauvegardés.")


def _render_styling_config():
    """Configuration de la charte graphique."""
    config = st.session_state.project_state.config
    styling = config.get("styling", {})

    col1, col2 = st.columns(2)
    with col1:
        primary_color = st.color_picker("Couleur principale", value=styling.get("primary_color", "#F0C441"))
        font_title = st.text_input("Police des titres", value=styling.get("font_title", "Calibri"))
        font_size_title = st.number_input("Taille des titres", 10, 36, value=styling.get("font_size_title", 16))
    with col2:
        secondary_color = st.color_picker("Couleur secondaire", value=styling.get("secondary_color", "#4E4E50"))
        font_body = st.text_input("Police du texte", value=styling.get("font_body", "Calibri"))
        font_size_body = st.number_input("Taille du texte", 8, 18, value=styling.get("font_size_body", 11))

    if st.button("Sauvegarder la charte graphique"):
        config["styling"] = {
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "font_title": font_title,
            "font_body": font_body,
            "font_size_title": font_size_title,
            "font_size_body": font_size_body,
            "margin_top_cm": styling.get("margin_top_cm", 2.5),
            "margin_bottom_cm": styling.get("margin_bottom_cm", 2.5),
            "margin_left_cm": styling.get("margin_left_cm", 2.5),
            "margin_right_cm": styling.get("margin_right_cm", 2.5),
        }
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Charte graphique sauvegardée.")


def _render_rag_config():
    """Configuration du RAG et de la génération conditionnelle."""
    config = st.session_state.project_state.config

    st.markdown("**Paramètres RAG (Retrieval-Augmented Generation)**")

    rag_top_k = st.number_input(
        "Nombre de blocs RAG par section (top-K)",
        min_value=1, max_value=20,
        value=config.get("rag_top_k", 7),
        help="Nombre de blocs de corpus les plus pertinents injectés dans le prompt.",
    )

    st.markdown("**Génération conditionnelle**")

    conditional_enabled = st.checkbox(
        "Activer la génération conditionnelle",
        value=config.get("conditional_generation_enabled", True),
        help="Évalue la couverture du corpus avant de générer chaque section.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sufficient = st.number_input(
            "Seuil suffisant",
            min_value=0.0, max_value=1.0,
            value=config.get("coverage_sufficient_threshold", 0.5),
            step=0.05,
        )
    with col2:
        insufficient = st.number_input(
            "Seuil insuffisant",
            min_value=0.0, max_value=1.0,
            value=config.get("coverage_insufficient_threshold", 0.3),
            step=0.05,
        )
    with col3:
        min_blocks = st.number_input(
            "Blocs minimum",
            min_value=0, max_value=10,
            value=config.get("coverage_min_blocks", 3),
        )

    if st.button("Sauvegarder la configuration RAG"):
        config["rag_top_k"] = rag_top_k
        config["conditional_generation_enabled"] = conditional_enabled
        config["coverage_sufficient_threshold"] = sufficient
        config["coverage_insufficient_threshold"] = insufficient
        config["coverage_min_blocks"] = min_blocks
        st.session_state.project_state.config = config
        _save_state(st.session_state.project_state)
        st.success("Configuration RAG sauvegardée.")


def _render_config_export_import():
    """Export et import de la configuration du projet."""
    import yaml

    config = st.session_state.project_state.config

    st.markdown("**Exporter la configuration**")
    config_yaml = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)

    st.download_button(
        "Télécharger la configuration (YAML)",
        data=config_yaml,
        file_name=f"config_{st.session_state.project_state.name}.yaml",
        mime="text/yaml",
    )

    st.markdown("**Importer une configuration**")
    uploaded = st.file_uploader("Charger un fichier YAML", type=["yaml", "yml"], key="config_import")
    if uploaded is not None:
        try:
            imported_config = yaml.safe_load(uploaded.read())
            if isinstance(imported_config, dict):
                if st.button("Appliquer la configuration importée"):
                    config.update(imported_config)
                    st.session_state.project_state.config = config
                    _save_state(st.session_state.project_state)
                    st.success("Configuration importée et appliquée.")
                    st.rerun()
            else:
                st.error("Le fichier ne contient pas une configuration valide.")
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
