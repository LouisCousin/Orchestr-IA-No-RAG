"""Personas cibles pour adapter le style de génération.

Phase 3 : définit le public cible du document pour adapter le style,
le vocabulaire et le niveau de détail.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from src.providers.base import BaseProvider
from src.utils.file_utils import save_json, load_json

logger = logging.getLogger("orchestria")

EXPERTISE_LEVELS = ("novice", "intermédiaire", "expert")
REGISTERS = ("formel", "courant", "technique")

SUGGEST_PERSONAS_PROMPT = """À partir de l'objectif du projet et du plan ci-dessous, propose 2-3 personas cibles pertinents.

═══ OBJECTIF DU PROJET ═══
{objective}

═══ PLAN DU DOCUMENT ═══
{plan_content}

═══ INSTRUCTIONS ═══
Pour chaque persona, retourne EXACTEMENT le format JSON suivant :
{{
  "personas": [
    {{
      "name": "<nom descriptif du persona>",
      "profile": "<description du profil type en 1-3 phrases>",
      "expertise_level": "novice|intermédiaire|expert",
      "expectations": "<ce que le persona attend du document>",
      "register": "formel|courant|technique",
      "sensitivities": "<sujets sensibles ou à traiter avec précaution>",
      "preferred_formats": ["<format préféré>", ...]
    }},
    ...
  ]
}}"""


class PersonaEngine:
    """Gère les personas cibles du projet."""

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        enabled: bool = False,
    ):
        self.project_dir = project_dir
        self.enabled = enabled
        self._personas: list[dict] = []
        self._primary_id: Optional[str] = None
        self._section_assignments: dict[str, str] = {}  # section_id → persona_id
        if project_dir:
            self._load()

    def _personas_path(self) -> Optional[Path]:
        if self.project_dir:
            return self.project_dir / "personas.json"
        return None

    def _load(self) -> None:
        """Charge les personas depuis le disque."""
        path = self._personas_path()
        if path and path.exists():
            try:
                data = load_json(path)
                self._personas = data.get("personas", [])
                self._primary_id = data.get("primary_id")
                self._section_assignments = data.get("section_assignments", {})
            except Exception as e:
                logger.warning(f"Erreur chargement personas : {e}")
                self._personas = []

    def _save(self) -> None:
        """Sauvegarde les personas sur disque."""
        path = self._personas_path()
        if path:
            save_json(path, {
                "personas": self._personas,
                "primary_id": self._primary_id,
                "section_assignments": self._section_assignments,
            })

    # ── CRUD ──

    def create(
        self,
        name: str,
        profile: str,
        expertise_level: str = "intermédiaire",
        expectations: str = "",
        register: str = "formel",
        sensitivities: str = "",
        preferred_formats: Optional[list[str]] = None,
    ) -> str:
        """Crée un nouveau persona.

        Returns:
            L'identifiant UUID du persona.
        """
        persona_id = str(uuid.uuid4())
        persona = {
            "id": persona_id,
            "name": name,
            "profile": profile,
            "expertise_level": expertise_level if expertise_level in EXPERTISE_LEVELS else "intermédiaire",
            "expectations": expectations,
            "register": register if register in REGISTERS else "formel",
            "sensitivities": sensitivities,
            "preferred_formats": preferred_formats or [],
        }
        self._personas.append(persona)
        # If first persona, set as primary
        if len(self._personas) == 1:
            self._primary_id = persona_id
        self._save()
        return persona_id

    def get(self, persona_id: str) -> Optional[dict]:
        """Retourne un persona par son identifiant."""
        for p in self._personas:
            if p["id"] == persona_id:
                return dict(p)
        return None

    def update(self, persona_id: str, **fields) -> Optional[dict]:
        """Met à jour un persona."""
        for p in self._personas:
            if p["id"] == persona_id:
                for key, value in fields.items():
                    if key != "id":
                        p[key] = value
                self._save()
                return dict(p)
        return None

    def delete(self, persona_id: str) -> bool:
        """Supprime un persona."""
        for i, p in enumerate(self._personas):
            if p["id"] == persona_id:
                self._personas.pop(i)
                if self._primary_id == persona_id:
                    self._primary_id = self._personas[0]["id"] if self._personas else None
                # Clean up section assignments
                self._section_assignments = {
                    k: v for k, v in self._section_assignments.items()
                    if v != persona_id
                }
                self._save()
                return True
        return False

    def list_personas(self) -> list[dict]:
        """Liste tous les personas."""
        return list(self._personas)

    # ── Assignation ──

    def set_primary(self, persona_id: str) -> bool:
        """Désigne le persona principal du projet."""
        if any(p["id"] == persona_id for p in self._personas):
            self._primary_id = persona_id
            self._save()
            return True
        return False

    def get_primary(self) -> Optional[dict]:
        """Retourne le persona principal."""
        if self._primary_id:
            return self.get(self._primary_id)
        return self._personas[0] if self._personas else None

    def assign_to_section(self, section_id: str, persona_id: str) -> None:
        """Assigne un persona à une section spécifique."""
        self._section_assignments[section_id] = persona_id
        self._save()

    def get_persona_for_section(self, section_id: str) -> Optional[dict]:
        """Retourne le persona actif pour une section.

        Si un persona est assigné à la section, il est retourné.
        Sinon, le persona principal est retourné.
        """
        assigned_id = self._section_assignments.get(section_id)
        if assigned_id:
            persona = self.get(assigned_id)
            if persona:
                return persona
        return self.get_primary()

    # ── Suggestion par IA ──

    def suggest_personas(
        self,
        plan,
        objective: str,
        provider: BaseProvider,
        model: Optional[str] = None,
    ) -> list[dict]:
        """Suggère des personas via l'IA.

        Args:
            plan: NormalizedPlan.
            objective: Objectif du projet.
            provider: Fournisseur IA.
            model: Modèle à utiliser.

        Returns:
            Liste de personas suggérés.
        """
        plan_content = "\n".join(
            f"- {s.id} {s.title}" + (f" : {s.description}" if s.description else "")
            for s in plan.sections
        )

        prompt = SUGGEST_PERSONAS_PROMPT.format(
            objective=objective,
            plan_content=plan_content,
        )

        try:
            model = model or provider.get_default_model()
            response = provider.generate(
                prompt=prompt,
                system_prompt="Tu es un expert en communication documentaire. Retourne uniquement du JSON valide.",
                model=model,
                temperature=0.5,
                max_tokens=1500,
            )
            return self._parse_personas(response.content)
        except Exception as e:
            logger.warning(f"Suggestion de personas échouée : {e}")
            return []

    @staticmethod
    def _parse_personas(response_text: str) -> list[dict]:
        """Parse la réponse JSON de l'IA."""
        text = response_text.strip()
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("personas", [])
            except (json.JSONDecodeError, ValueError):
                pass
        return []

    # ── Injection dans les prompts ──

    def format_for_prompt(self, persona: Optional[dict] = None) -> str:
        """Formate un persona pour injection dans un prompt.

        Args:
            persona: Dict du persona. Si None, utilise le persona principal.

        Returns:
            Bloc de texte pour le prompt, ou chaîne vide.
        """
        if not self.enabled:
            return ""

        if persona is None:
            persona = self.get_primary()
        if not persona:
            return ""

        lines = [
            f"Tu rédiges pour le public suivant : {persona.get('profile', '')}.",
            f"Niveau d'expertise : {persona.get('expertise_level', 'intermédiaire')}.",
            f"Registre : {persona.get('register', 'formel')}.",
        ]
        if persona.get("expectations"):
            lines.append(f"Attentes : {persona['expectations']}.")
        if persona.get("sensitivities"):
            lines.append(f"Sujets sensibles : {persona['sensitivities']}.")
        if persona.get("preferred_formats"):
            lines.append(f"Formats préférés : {', '.join(persona['preferred_formats'])}.")

        return "\n".join(lines)
