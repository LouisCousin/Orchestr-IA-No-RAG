"""Normalisation du plan structurel du livrable."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.text_extractor import extract

logger = logging.getLogger("orchestria")


@dataclass
class PlanSection:
    """Section du plan normalisé."""
    id: str
    title: str
    level: int  # 1 = titre principal, 2 = sous-section, etc.
    parent_id: Optional[str] = None
    description: str = ""
    page_budget: Optional[float] = None
    generated_content: Optional[str] = None
    status: str = "pending"  # "pending", "generating", "generated", "validated"
    metadata: dict = field(default_factory=dict)


@dataclass
class NormalizedPlan:
    """Plan normalisé : arbre de sections."""
    sections: list[PlanSection] = field(default_factory=list)
    title: str = ""
    objective: str = ""
    raw_text: str = ""

    def get_section(self, section_id: str) -> Optional[PlanSection]:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None

    def get_root_sections(self) -> list[PlanSection]:
        return [s for s in self.sections if s.level == 1]

    def get_children(self, section_id: str) -> list[PlanSection]:
        return [s for s in self.sections if s.parent_id == section_id]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "objective": self.objective,
            "raw_text": self.raw_text,
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "level": s.level,
                    "parent_id": s.parent_id,
                    "description": s.description,
                    "page_budget": s.page_budget,
                    "status": s.status,
                }
                for s in self.sections
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizedPlan":
        plan = cls(
            title=data.get("title", ""),
            objective=data.get("objective", ""),
            raw_text=data.get("raw_text", ""),
        )
        for s in data.get("sections", []):
            plan.sections.append(PlanSection(
                id=s["id"],
                title=s["title"],
                level=s["level"],
                parent_id=s.get("parent_id"),
                description=s.get("description", ""),
                page_budget=s.get("page_budget"),
                status=s.get("status", "pending"),
            ))
        return plan


class PlanParser:
    """Parseur et normalisateur de plan."""

    # Patterns de détection de niveaux de titres
    HEADING_PATTERNS = [
        # Numérotation décimale avec sous-niveaux : 1.1 / 1.1.1 / 1.2.3
        (r"^(\d+(?:\.\d+)+)\s+(.+)", "decimal"),
        # Numérotation décimale simple : 1. / 2. / 3)
        (r"^(\d+)\s*[.:\-–—)]\s+(.+)", "decimal"),
        # Markdown : # / ## / ###
        (r"^(#{1,6})\s+(.+)", "markdown"),
        # Numérotation romaine : I. / II. / III.
        (r"^((?:I{1,3}|IV|V(?:I{0,3})|IX|X(?:I{0,3})?))\s*[.:\-–—)]\s+(.+)", "roman"),
        # Lettres : A. / B. / a) / b)
        (r"^([A-Z])\s*[.)\-]\s+(.+)", "letter_upper"),
        (r"^([a-z])\s*[.)]\s+(.+)", "letter_lower"),
    ]

    def parse_file(self, file_path: Path) -> NormalizedPlan:
        """Parse un fichier plan et retourne un plan normalisé."""
        result = extract(file_path)
        if result.status == "failed":
            raise ValueError(f"Impossible d'extraire le texte du plan : {result.error_message}")
        return self.parse_text(result.text)

    def parse_text(self, text: str) -> NormalizedPlan:
        """Parse un texte brut de plan et le normalise."""
        plan = NormalizedPlan(raw_text=text)
        lines = text.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            return plan

        # Détecter le titre du document (première ligne significative)
        plan.title = lines[0] if lines else ""

        sections = self._detect_sections(lines)
        if sections:
            plan.sections = sections
        else:
            # Fallback : chaque ligne non vide est une section de niveau 1
            plan.sections = self._fallback_parse(lines)

        self._assign_ids(plan)
        self._assign_parent_ids(plan)
        return plan

    def _detect_sections(self, lines: list[str]) -> list[PlanSection]:
        """Détecte et parse les sections à partir des patterns de titres."""
        sections = []
        current_description_lines = []

        for line in lines:
            matched = False
            for pattern, pattern_type in self.HEADING_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    # Sauvegarder la description de la section précédente
                    if sections and current_description_lines:
                        sections[-1].description = " ".join(current_description_lines)
                        current_description_lines = []

                    level = self._determine_level(match.group(1), pattern_type)
                    title = match.group(2).strip()

                    sections.append(PlanSection(
                        id="",  # Assigné plus tard
                        title=title,
                        level=level,
                    ))
                    matched = True
                    break

            if not matched and sections:
                current_description_lines.append(line)

        # Dernière section
        if sections and current_description_lines:
            sections[-1].description = " ".join(current_description_lines)

        return sections

    def _determine_level(self, marker: str, pattern_type: str) -> int:
        """Détermine le niveau hiérarchique d'un marqueur."""
        if pattern_type == "decimal":
            return marker.count(".") + 1
        elif pattern_type == "markdown":
            return len(marker)
        elif pattern_type == "roman":
            return 1
        elif pattern_type == "letter_upper":
            return 2
        elif pattern_type == "letter_lower":
            return 3
        return 1

    def _fallback_parse(self, lines: list[str]) -> list[PlanSection]:
        """Parse de secours : chaque ligne est une section de niveau 1."""
        sections = []
        for line in lines:
            if line.strip():
                sections.append(PlanSection(id="", title=line.strip(), level=1))
        return sections

    def _assign_ids(self, plan: NormalizedPlan) -> None:
        """Assigne des identifiants uniques à chaque section."""
        counters = {}
        for section in plan.sections:
            level = section.level
            counters[level] = counters.get(level, 0) + 1
            # Réinitialiser les compteurs des niveaux inférieurs
            for l in list(counters.keys()):
                if l > level:
                    del counters[l]

            parts = [str(counters.get(l, 1)) for l in range(1, level + 1)]
            section.id = ".".join(parts)

    def _assign_parent_ids(self, plan: NormalizedPlan) -> None:
        """Assigne les parent_id basés sur la hiérarchie des niveaux."""
        stack: list[PlanSection] = []
        for section in plan.sections:
            while stack and stack[-1].level >= section.level:
                stack.pop()
            if stack:
                section.parent_id = stack[-1].id
            stack.append(section)

    def distribute_page_budget(self, plan: NormalizedPlan, target_pages: Optional[float]) -> None:
        """Répartit le budget de pages entre les sections proportionnellement."""
        if target_pages is None or target_pages <= 0:
            return

        root_sections = plan.get_root_sections()
        if not root_sections:
            return

        # Répartition proportionnelle (uniforme par défaut)
        # Les sections avec plus de sous-sections reçoivent plus de pages
        weights = []
        for section in root_sections:
            children = plan.get_children(section.id)
            weight = max(1, len(children) + 1)
            weights.append(weight)

        total_weight = sum(weights)
        for section, weight in zip(root_sections, weights):
            section.page_budget = round(target_pages * weight / total_weight, 1)

            # Distribuer aux enfants
            children = plan.get_children(section.id)
            if children and section.page_budget:
                child_budget = section.page_budget / len(children)
                for child in children:
                    child.page_budget = round(child_budget, 1)
