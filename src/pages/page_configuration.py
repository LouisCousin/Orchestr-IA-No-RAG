"""Page de configuration — Paramètres du projet et du fournisseur IA."""

import os
import streamlit as st

from src.providers.openai_provider import OpenAIProvider
from src.core.checkpoint_manager import CheckpointConfig
from src.core.cost_tracker import CostTracker


def render():
    st.title("Configuration")
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    _render_api_config()
    st.markdown("---")
    _render_model_config()
    st.markdown("---")

    with st.expander("Points de controle (HITL)", expanded=False):
        _render_checkpoint_config()

    with st.expander("Charte graphique", expanded=False):
        _render_styling_config()


def _render_api_config():
    """Configuration de la clé API."""
    st.subheader("Fournisseur IA")

    api_key_env = os.environ.get("OPENAI_API_KEY", "")
    has_env_key = bool(api_key_env and api_key_env != "sk-your-openai-api-key-here")

    if has_env_key:
        st.success("Clé API OpenAI détectée dans le fichier .env")
        masked = api_key_env[:8] + "..." + api_key_env[-4:] if len(api_key_env) > 12 else "***"
        st.code(masked)
    else:
        st.warning("Aucune clé API détectée. Saisissez votre clé ci-dessous ou configurez le fichier .env")

    api_key_input = st.text_input(
        "Clé API OpenAI (session uniquement)",
        type="password",
        help="La clé est stockée en session uniquement et n'est pas sauvegardée sur le disque.",
    )

    if st.button("Valider la connexion", type="primary"):
        key = api_key_input or api_key_env
        if not key:
            st.error("Veuillez saisir une clé API.")
            return

        provider = OpenAIProvider(api_key=key)
        if provider.is_available():
            st.session_state.provider = provider
            st.session_state.cost_tracker = CostTracker()
            st.success("Connexion OpenAI validée !")
        else:
            st.error("Clé API invalide ou vide.")


def _render_model_config():
    """Configuration du modèle IA."""
    st.subheader("Modèle de génération")

    config = st.session_state.project_state.config

    provider = st.session_state.get("provider")
    if provider:
        models = provider.list_models()
    else:
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    current_model = config.get("model", "gpt-4o")
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

    if st.button("Sauvegarder la configuration du modèle"):
        config["model"] = model
        config["temperature"] = temperature
        config["max_tokens"] = max_tokens
        st.session_state.project_state.config = config
        st.success("Configuration sauvegardée.")


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
        st.success("Charte graphique sauvegardée.")
