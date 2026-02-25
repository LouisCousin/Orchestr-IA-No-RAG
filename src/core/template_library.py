"""Bibliothèque de templates de prompts réutilisables.

Phase 3 : permet de sauvegarder, gérer et réutiliser des templates
de prompts entre projets.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir

logger = logging.getLogger("orchestria")

TEMPLATES_DIR = ROOT_DIR / "templates"
LIBRARY_PATH = TEMPLATES_DIR / "library.json"
PRESETS_DIR = TEMPLATES_DIR / "presets"


class TemplateVariableError(Exception):
    """Erreur levée quand une variable requise est manquante."""
    pass


# 5 templates préconfigurés
DEFAULT_TEMPLATES = [
    {
        "name": "Synthèse académique",
        "description": "Pour résumer des articles de recherche et produire une synthèse académique structurée.",
        "tags": ["académique", "synthèse", "recherche"],
        "content": (
            "Rédige une synthèse académique sur le thème « {sujet} ».\n"
            "La synthèse doit couvrir les points suivants : {points_cles}.\n"
            "Niveau de détail : {niveau_detail}.\n"
            "Public cible : {public_cible}."
        ),
        "variables": [
            {"name": "sujet", "description": "Sujet principal de la synthèse"},
            {"name": "points_cles", "description": "Points clés à couvrir", "default": "les concepts principaux, les méthodologies, les résultats clés"},
            {"name": "niveau_detail", "description": "Niveau de détail attendu", "default": "approfondi"},
            {"name": "public_cible", "description": "Public visé", "default": "chercheurs et étudiants gradués"},
        ],
    },
    {
        "name": "Analyse comparative",
        "description": "Pour comparer des approches, solutions ou méthodologies.",
        "tags": ["analyse", "comparaison", "stratégie"],
        "content": (
            "Produis une analyse comparative de {elements_compares}.\n"
            "Critères de comparaison : {criteres}.\n"
            "Format de présentation : {format_presentation}.\n"
            "Objectif de l'analyse : {objectif}."
        ),
        "variables": [
            {"name": "elements_compares", "description": "Éléments à comparer"},
            {"name": "criteres", "description": "Critères de comparaison", "default": "avantages, inconvénients, coûts, faisabilité"},
            {"name": "format_presentation", "description": "Format souhaité", "default": "tableau comparatif suivi d'une analyse narrative"},
            {"name": "objectif", "description": "Objectif de l'analyse", "default": "identifier la meilleure option"},
        ],
    },
    {
        "name": "Section didactique",
        "description": "Pour rédiger du contenu pédagogique accessible et progressif.",
        "tags": ["pédagogie", "formation", "didactique"],
        "content": (
            "Rédige une section pédagogique sur « {concept} ».\n"
            "Niveau du public : {niveau_public}.\n"
            "Inclure : {elements_pedagogiques}.\n"
            "Style : {style_redaction}."
        ),
        "variables": [
            {"name": "concept", "description": "Concept ou sujet à enseigner"},
            {"name": "niveau_public", "description": "Niveau du public cible", "default": "intermédiaire"},
            {"name": "elements_pedagogiques", "description": "Éléments pédagogiques à inclure", "default": "définitions, exemples concrets, exercices pratiques"},
            {"name": "style_redaction", "description": "Style de rédaction", "default": "accessible et progressif"},
        ],
    },
    {
        "name": "Résumé exécutif",
        "description": "Pour produire un résumé destiné aux décideurs.",
        "tags": ["résumé", "exécutif", "décision"],
        "content": (
            "Produis un résumé exécutif sur « {sujet} ».\n"
            "Points clés à synthétiser : {points_cles}.\n"
            "Recommandations attendues : {recommandations}.\n"
            "Longueur : {longueur}."
        ),
        "variables": [
            {"name": "sujet", "description": "Sujet du résumé exécutif"},
            {"name": "points_cles", "description": "Points clés à couvrir", "default": "contexte, constats, enjeux"},
            {"name": "recommandations", "description": "Type de recommandations", "default": "actions concrètes priorisées"},
            {"name": "longueur", "description": "Longueur cible", "default": "1 à 2 pages"},
        ],
    },
    {
        "name": "Section factuelle",
        "description": "Pour rédiger à partir de données chiffrées et factuelles.",
        "tags": ["données", "factuel", "statistiques"],
        "content": (
            "Rédige une section factuelle sur « {sujet} ».\n"
            "Données à exploiter : {donnees}.\n"
            "Mise en forme : {mise_en_forme}.\n"
            "Interprétation attendue : {interpretation}."
        ),
        "variables": [
            {"name": "sujet", "description": "Sujet de la section"},
            {"name": "donnees", "description": "Données à exploiter", "default": "les données chiffrées du corpus"},
            {"name": "mise_en_forme", "description": "Format de présentation", "default": "tableaux et graphiques commentés"},
            {"name": "interpretation", "description": "Niveau d'interprétation", "default": "analyse descriptive et tendances"},
        ],
    },
]


class TemplateLibrary:
    """Bibliothèque de templates de prompts."""

    def __init__(self, library_path: Optional[Path] = None):
        self.library_path = library_path or LIBRARY_PATH
        self._templates: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Charge les templates depuis le disque."""
        if self.library_path.exists():
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    self._templates = json.load(f)
            except Exception as e:
                logger.warning(f"Erreur chargement bibliothèque de templates : {e}")
                self._templates = []
        else:
            self._templates = []
            self._init_default_templates()

    def _save(self) -> None:
        """Sauvegarde les templates sur disque."""
        ensure_dir(self.library_path.parent)
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self._templates, f, ensure_ascii=False, indent=2)

    def _init_default_templates(self) -> None:
        """Initialise les templates préconfigurés."""
        for tpl_data in DEFAULT_TEMPLATES:
            self.create(
                name=tpl_data["name"],
                description=tpl_data["description"],
                content=tpl_data["content"],
                tags=tpl_data["tags"],
                variables=tpl_data["variables"],
            )

    # ── CRUD ──

    def create(
        self,
        name: str,
        content: str,
        description: str = "",
        tags: Optional[list[str]] = None,
        variables: Optional[list[dict]] = None,
        source_project: Optional[str] = None,
    ) -> str:
        """Crée un nouveau template.

        Args:
            name: Nom court du template.
            content: Texte du template avec variables {variable}.
            description: Description de l'usage prévu.
            tags: Tags de catégorisation.
            variables: Liste de variables attendues.
            source_project: Nom du projet d'origine.

        Returns:
            L'identifiant UUID du template créé.

        Raises:
            ValueError: Si un template avec le même nom existe déjà.
        """
        # Vérifier l'unicité du nom
        if any(t["name"].lower() == name.lower() for t in self._templates):
            raise ValueError(f"Un template nommé '{name}' existe déjà")

        now = datetime.now().isoformat()
        template = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "tags": tags or [],
            "content": content,
            "variables": variables or [],
            "created_at": now,
            "updated_at": now,
            "usage_count": 0,
            "source_project": source_project,
        }
        self._templates.append(template)
        self._save()
        return template["id"]

    def get(self, template_id: str) -> Optional[dict]:
        """Retourne un template par son identifiant."""
        for t in self._templates:
            if t["id"] == template_id:
                return dict(t)
        return None

    def list(self, tags: Optional[list[str]] = None, search: Optional[str] = None) -> list[dict]:
        """Liste les templates avec filtrage optionnel.

        Args:
            tags: Filtrer par tags (tous les tags doivent être présents).
            search: Recherche textuelle dans le nom et la description.

        Returns:
            Liste de templates correspondants.
        """
        results = list(self._templates)

        if tags:
            tags_lower = [t.lower() for t in tags]
            results = [
                t for t in results
                if all(tag in [tt.lower() for tt in t.get("tags", [])] for tag in tags_lower)
            ]

        if search:
            search_lower = search.lower()
            results = [
                t for t in results
                if search_lower in t.get("name", "").lower()
                or search_lower in t.get("description", "").lower()
            ]

        return results

    def update(self, template_id: str, **fields) -> Optional[dict]:
        """Met à jour les champs spécifiés d'un template.

        Returns:
            Le template mis à jour, ou None si non trouvé.
        """
        for i, t in enumerate(self._templates):
            if t["id"] == template_id:
                for key, value in fields.items():
                    if key in ("id", "created_at"):
                        continue  # Protéger ces champs
                    t[key] = value
                t["updated_at"] = datetime.now().isoformat()
                self._save()
                return dict(t)
        return None

    def delete(self, template_id: str) -> bool:
        """Supprime un template.

        Returns:
            True si supprimé, False si non trouvé.
        """
        for i, t in enumerate(self._templates):
            if t["id"] == template_id:
                self._templates.pop(i)
                self._save()
                return True
        return False

    def duplicate(self, template_id: str, new_name: str) -> Optional[str]:
        """Duplique un template avec un nouveau nom.

        Returns:
            L'identifiant du nouveau template, ou None si original non trouvé.
        """
        original = self.get(template_id)
        if not original:
            return None

        return self.create(
            name=new_name,
            content=original["content"],
            description=original.get("description", ""),
            tags=original.get("tags", []),
            variables=original.get("variables", []),
            source_project=original.get("source_project"),
        )

    def export_template(self, template_id: str) -> Optional[dict]:
        """Exporte un template en JSON autonome."""
        return self.get(template_id)

    def import_template(self, data: dict) -> str:
        """Importe un template depuis un JSON.

        Gère les conflits de noms en ajoutant un suffixe.

        Returns:
            L'identifiant du template importé.
        """
        name = data.get("name", "Template importé")
        # Handle name conflicts
        original_name = name
        counter = 1
        while any(t["name"].lower() == name.lower() for t in self._templates):
            name = f"{original_name} ({counter})"
            counter += 1

        return self.create(
            name=name,
            content=data.get("content", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            variables=data.get("variables", []),
            source_project=data.get("source_project"),
        )

    # ── Résolution des variables ──

    def resolve(self, template_id: str, variables: Optional[dict] = None) -> str:
        """Remplace les variables dans le contenu du template.

        Args:
            template_id: Identifiant du template.
            variables: Dict {nom_variable: valeur}.

        Returns:
            Le texte avec les variables résolues.

        Raises:
            ValueError: Si le template n'existe pas.
            TemplateVariableError: Si une variable requise est manquante.
        """
        template = self.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} non trouvé")

        variables = variables or {}
        content = template["content"]
        template_vars = {v["name"]: v for v in template.get("variables", [])}

        # Check for missing required variables
        import re
        used_vars = set(re.findall(r'\{(\w+)\}', content))

        for var_name in used_vars:
            if var_name in variables:
                continue
            var_def = template_vars.get(var_name, {})
            if "default" in var_def:
                variables[var_name] = var_def["default"]
            else:
                raise TemplateVariableError(
                    f"Variable '{var_name}' requise mais non fournie et sans valeur par défaut"
                )

        # Replace variables
        for var_name, value in variables.items():
            content = content.replace(f"{{{var_name}}}", str(value))

        # Increment usage count
        self._increment_usage(template_id)

        return content

    def _increment_usage(self, template_id: str) -> None:
        """Incrémente le compteur d'utilisation."""
        for t in self._templates:
            if t["id"] == template_id:
                t["usage_count"] = t.get("usage_count", 0) + 1
                self._save()
                break
