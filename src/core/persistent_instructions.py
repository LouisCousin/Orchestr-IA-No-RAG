"""Instructions persistantes de projet — système hiérarchique structuré.

Phase 3 : remplace la chaîne simple persistent_instructions par un
système hiérarchique à 3 niveaux et 4 catégories.
"""

import logging
from pathlib import Path
from typing import Optional

from src.utils.file_utils import save_json, load_json

logger = logging.getLogger("orchestria")

# Catégories d'instructions
CATEGORIES = ("style_ton", "contenu", "structure", "domaine_metier")

CATEGORY_LABELS = {
    "style_ton": "Style et ton",
    "contenu": "Contenu obligatoire/interdit",
    "structure": "Structure",
    "domaine_metier": "Domaine métier",
}

# Niveaux hiérarchiques
LEVELS = ("project", "context", "section")


class PersistentInstructions:
    """Gère les instructions persistantes hiérarchiques."""

    def __init__(self, project_dir: Optional[Path] = None):
        self.project_dir = project_dir
        self._instructions: dict = {
            "project": {},      # category → text
            "contexts": {},     # context_name → {category → text}
            "sections": {},     # section_id → {category → text}
            "section_contexts": {},  # section_id → context_name
        }
        if project_dir:
            self._load()

    def _instructions_path(self) -> Optional[Path]:
        if self.project_dir:
            return self.project_dir / "persistent_instructions.json"
        return None

    def _load(self) -> None:
        """Charge depuis le disque."""
        path = self._instructions_path()
        if path and path.exists():
            try:
                data = load_json(path)
                self._instructions = {
                    "project": data.get("project", {}),
                    "contexts": data.get("contexts", {}),
                    "sections": data.get("sections", {}),
                    "section_contexts": data.get("section_contexts", {}),
                }
            except Exception as e:
                logger.warning(f"Erreur chargement instructions persistantes : {e}")

    def _save(self) -> None:
        """Sauvegarde sur disque."""
        path = self._instructions_path()
        if path:
            save_json(path, self._instructions)

    # ── Setters ──

    def set_project_instruction(self, category: str, text: str) -> None:
        """Définit une instruction de niveau Projet.

        Args:
            category: Catégorie (style_ton, contenu, structure, domaine_metier).
            text: Texte de l'instruction.
        """
        if category not in CATEGORIES:
            logger.warning(f"Catégorie inconnue : {category}")
            return
        self._instructions["project"][category] = text
        self._save()

    def set_context_instruction(self, context_name: str, category: str, text: str) -> None:
        """Définit une instruction de niveau Contexte.

        Args:
            context_name: Nom du contexte (groupe de sections).
            category: Catégorie.
            text: Texte de l'instruction.
        """
        if category not in CATEGORIES:
            logger.warning(f"Catégorie inconnue : {category}")
            return
        if context_name not in self._instructions["contexts"]:
            self._instructions["contexts"][context_name] = {}
        self._instructions["contexts"][context_name][category] = text
        self._save()

    def set_section_instruction(self, section_id: str, category: str, text: str) -> None:
        """Définit une instruction de niveau Section.

        Args:
            section_id: Identifiant de la section.
            category: Catégorie.
            text: Texte de l'instruction.
        """
        if category not in CATEGORIES:
            logger.warning(f"Catégorie inconnue : {category}")
            return
        if section_id not in self._instructions["sections"]:
            self._instructions["sections"][section_id] = {}
        self._instructions["sections"][section_id][category] = text
        self._save()

    def assign_section_to_context(self, section_id: str, context_name: str) -> None:
        """Assigne une section à un contexte."""
        self._instructions["section_contexts"][section_id] = context_name
        self._save()

    # ── Getters ──

    def get_project_instructions(self) -> dict:
        """Retourne les instructions de niveau Projet."""
        return dict(self._instructions.get("project", {}))

    def get_context_instructions(self, context_name: str) -> dict:
        """Retourne les instructions d'un contexte."""
        return dict(self._instructions.get("contexts", {}).get(context_name, {}))

    def get_section_instructions(self, section_id: str) -> dict:
        """Retourne les instructions spécifiques d'une section."""
        return dict(self._instructions.get("sections", {}).get(section_id, {}))

    def list_contexts(self) -> list[str]:
        """Liste les contextes définis."""
        return list(self._instructions.get("contexts", {}).keys())

    # ── Résolution hiérarchique ──

    def get_instructions(self, section_id: str) -> str:
        """Résout les instructions hiérarchiquement pour une section.

        Priorité : Section > Contexte > Projet.

        Args:
            section_id: Identifiant de la section.

        Returns:
            Texte des instructions fusionnées, prêt pour le prompt.
        """
        resolved = {}

        # Niveau 1 : Projet
        for cat, text in self._instructions.get("project", {}).items():
            if text:
                resolved[cat] = text

        # Niveau 2 : Contexte (si la section est assignée)
        context_name = self._instructions.get("section_contexts", {}).get(section_id)
        if context_name:
            context_instructions = self._instructions.get("contexts", {}).get(context_name, {})
            for cat, text in context_instructions.items():
                if text:
                    # Detect conflicts
                    if cat in resolved and resolved[cat] != text:
                        logger.warning(
                            f"Conflit d'instruction ({cat}) : le contexte '{context_name}' "
                            f"remplace le niveau Projet pour la section {section_id}"
                        )
                    resolved[cat] = text

        # Niveau 3 : Section
        section_instructions = self._instructions.get("sections", {}).get(section_id, {})
        for cat, text in section_instructions.items():
            if text:
                if cat in resolved and resolved[cat] != text:
                    logger.warning(
                        f"Conflit d'instruction ({cat}) : le niveau Section "
                        f"remplace les niveaux supérieurs pour {section_id}"
                    )
                resolved[cat] = text

        if not resolved:
            return ""

        # Format for prompt
        parts = []
        for cat in CATEGORIES:
            if cat in resolved:
                label = CATEGORY_LABELS.get(cat, cat)
                parts.append(f"[{label}] {resolved[cat]}")

        return "\n".join(parts)

    def detect_conflicts(self, section_id: str) -> list[dict]:
        """Détecte les conflits entre niveaux pour une section.

        Returns:
            Liste de conflits détectés {category, project_text, context_text, section_text}.
        """
        conflicts = []
        project = self._instructions.get("project", {})
        context_name = self._instructions.get("section_contexts", {}).get(section_id)
        context = self._instructions.get("contexts", {}).get(context_name, {}) if context_name else {}
        section = self._instructions.get("sections", {}).get(section_id, {})

        for cat in CATEGORIES:
            texts = {}
            if cat in project and project[cat]:
                texts["project"] = project[cat]
            if cat in context and context[cat]:
                texts["context"] = context[cat]
            if cat in section and section[cat]:
                texts["section"] = section[cat]

            if len(texts) > 1:
                # Check if they differ
                unique_values = set(texts.values())
                if len(unique_values) > 1:
                    conflicts.append({
                        "category": cat,
                        "category_label": CATEGORY_LABELS.get(cat, cat),
                        "project_text": texts.get("project", ""),
                        "context_text": texts.get("context", ""),
                        "section_text": texts.get("section", ""),
                        "resolution": texts.get("section") or texts.get("context") or texts.get("project", ""),
                    })

        return conflicts

    def is_configured(self) -> bool:
        """Vérifie si des instructions sont configurées."""
        if self._instructions.get("project"):
            return True
        if self._instructions.get("contexts"):
            return True
        if self._instructions.get("sections"):
            return True
        return False

    def to_dict(self) -> dict:
        """Sérialise les instructions."""
        return dict(self._instructions)

    @classmethod
    def from_dict(cls, data: dict, project_dir: Optional[Path] = None) -> "PersistentInstructions":
        """Désérialise les instructions."""
        inst = cls(project_dir=None)  # Don't load from disk
        inst.project_dir = project_dir
        inst._instructions = {
            "project": data.get("project", {}),
            "contexts": data.get("contexts", {}),
            "sections": data.get("sections", {}),
            "section_contexts": data.get("section_contexts", {}),
        }
        return inst
