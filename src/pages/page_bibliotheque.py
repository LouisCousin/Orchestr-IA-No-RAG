"""Page Bibliothèque — Gestion des templates de prompts.

Phase 3 : interface de gestion complète des templates avec CRUD,
filtrage et aperçu.
"""

import json
import streamlit as st

from src.core.template_library import TemplateLibrary, TemplateVariableError


def _get_library() -> TemplateLibrary:
    """Récupère ou initialise la bibliothèque de templates."""
    if "template_library" not in st.session_state:
        st.session_state.template_library = TemplateLibrary()
    return st.session_state.template_library


def render():
    """Affiche la page de gestion des templates."""
    st.title("Bibliothèque de templates")
    st.markdown("Gérez vos templates de prompts réutilisables entre projets.")

    # B12: initialize session state key before access
    if "selected_template_id" not in st.session_state:
        st.session_state.selected_template_id = None

    library = _get_library()

    # Layout : colonne gauche (liste) + colonne droite (détail)
    col_left, col_right = st.columns([3, 7])

    with col_left:
        _render_template_list(library)

    with col_right:
        _render_template_detail(library)


def _render_template_list(library: TemplateLibrary):
    """Colonne gauche : liste des templates avec filtres."""
    st.subheader("Templates")

    # Filtres
    search = st.text_input("Rechercher", key="tpl_search", placeholder="Nom ou description...")

    # Get all tags for filter
    all_templates = library.list()
    all_tags = sorted(set(tag for t in all_templates for tag in t.get("tags", [])))
    selected_tags = st.multiselect("Filtrer par tags", all_tags, key="tpl_tags")

    # Liste filtrée
    templates = library.list(
        tags=selected_tags if selected_tags else None,
        search=search if search else None,
    )

    if not templates:
        st.info("Aucun template trouvé.")
    else:
        for t in templates:
            label = t["name"]
            usage = t.get("usage_count", 0)
            if usage > 0:
                label += f" ({usage}x)"
            if st.button(label, key=f"tpl_select_{t['id']}", use_container_width=True):
                st.session_state.selected_template_id = t["id"]

    st.divider()

    # Bouton Nouveau
    if st.button("Nouveau template", use_container_width=True, type="primary"):
        st.session_state.selected_template_id = "__new__"

    # Import
    with st.expander("Importer un template"):
        uploaded = st.file_uploader("Fichier JSON", type=["json"], key="tpl_import")
        if uploaded and st.button("Importer"):
            try:
                data = json.loads(uploaded.read().decode("utf-8"))
                new_id = library.import_template(data)
                st.success(f"Template importé (ID: {new_id[:8]}...)")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur import : {e}")


def _render_template_detail(library: TemplateLibrary):
    """Colonne droite : détail et édition du template sélectionné."""
    selected_id = st.session_state.get("selected_template_id")

    if not selected_id:
        st.info("Sélectionnez un template dans la liste ou créez-en un nouveau.")
        return

    if selected_id == "__new__":
        _render_new_template_form(library)
        return

    template = library.get(selected_id)
    if not template:
        st.warning("Template non trouvé.")
        return

    st.subheader(template["name"])
    st.caption(f"ID: {template['id'][:8]}... | Créé: {template.get('created_at', 'N/A')[:10]} | Utilisé: {template.get('usage_count', 0)} fois")

    # Description
    description = st.text_input("Description", value=template.get("description", ""), key="tpl_desc")

    # Tags
    current_tags = template.get("tags", [])
    tags_str = st.text_input("Tags (séparés par des virgules)", value=", ".join(current_tags), key="tpl_tags_edit")
    new_tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    # Contenu
    content = st.text_area("Contenu du template", value=template.get("content", ""), height=200, key="tpl_content")

    # Variables
    st.markdown("**Variables**")
    variables = template.get("variables", [])
    edited_vars = []
    for i, var in enumerate(variables):
        with st.container():
            c1, c2, c3 = st.columns([2, 3, 2])
            with c1:
                name = st.text_input("Nom", value=var.get("name", ""), key=f"var_name_{selected_id}_{i}")
            with c2:
                desc = st.text_input("Description", value=var.get("description", ""), key=f"var_desc_{selected_id}_{i}")
            with c3:
                default = st.text_input("Défaut", value=var.get("default", ""), key=f"var_def_{selected_id}_{i}")
            edited_vars.append({
                "name": name,
                "description": desc,
                **({"default": default} if default else {}),
            })

    # Actions
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Sauvegarder", type="primary"):
            library.update(
                selected_id,
                description=description,
                tags=new_tags,
                content=content,
                variables=edited_vars,
            )
            st.success("Template sauvegardé.")
            st.rerun()

    with col2:
        dup_name = st.text_input("Nom du doublon", key="tpl_dup_name", placeholder="Nom...")
        if dup_name and st.button("Dupliquer"):
            try:
                new_id = library.duplicate(selected_id, dup_name)
                st.success(f"Template dupliqué.")
                st.session_state.selected_template_id = new_id
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    with col3:
        if st.button("Exporter JSON"):
            data = library.export_template(selected_id)
            if data:
                st.download_button(
                    "Télécharger",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name=f"{template['name'].replace(' ', '_')}.json",
                    mime="application/json",
                )

    with col4:
        if st.button("Supprimer", type="secondary"):
            library.delete(selected_id)
            st.session_state.selected_template_id = None
            st.rerun()

    # Aperçu résolu
    st.divider()
    st.markdown("**Aperçu du prompt résolu**")
    with st.expander("Résoudre les variables"):
        var_values = {}
        for i, var in enumerate(variables):
            val = st.text_input(
                f"{var['name']}",
                value=var.get("default", ""),
                key=f"resolve_{selected_id}_{i}_{var['name']}",
            )
            var_values[var["name"]] = val

        if st.button("Résoudre"):
            try:
                resolved = library.resolve(selected_id, var_values)
                st.code(resolved, language=None)
            except TemplateVariableError as e:
                st.error(str(e))


def _render_new_template_form(library: TemplateLibrary):
    """Formulaire de création d'un nouveau template."""
    st.subheader("Nouveau template")

    name = st.text_input("Nom", key="new_tpl_name")
    description = st.text_input("Description", key="new_tpl_desc")
    tags_str = st.text_input("Tags (séparés par des virgules)", key="new_tpl_tags")
    content = st.text_area("Contenu (utilisez {variable} pour les variables)", key="new_tpl_content", height=200)

    if st.button("Créer", type="primary"):
        if not name:
            st.error("Le nom est requis.")
            return
        if not content:
            st.error("Le contenu est requis.")
            return

        try:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            new_id = library.create(
                name=name,
                content=content,
                description=description,
                tags=tags,
            )
            st.success(f"Template créé : {name}")
            st.session_state.selected_template_id = new_id
            st.rerun()
        except ValueError as e:
            st.error(str(e))
