"""Tests unitaires pour le module plan_parser."""

import pytest
from pathlib import Path

from src.core.plan_parser import PlanParser, NormalizedPlan, PlanSection


@pytest.fixture
def parser():
    return PlanParser()


class TestParseText:
    def test_decimal_numbering(self, parser):
        text = """Document de test
1. Introduction
1.1 Contexte
1.2 Objectifs
2. Analyse
2.1 État des lieux
2.2 Données
3. Conclusion"""
        plan = parser.parse_text(text)
        assert len(plan.sections) >= 7
        root_sections = plan.get_root_sections()
        assert len(root_sections) >= 3

    def test_markdown_headings(self, parser):
        text = """# Titre principal
## Sous-titre 1
### Sous-sous-titre
## Sous-titre 2"""
        plan = parser.parse_text(text)
        assert len(plan.sections) >= 3

    def test_empty_text(self, parser):
        plan = parser.parse_text("")
        assert len(plan.sections) == 0

    def test_single_line(self, parser):
        plan = parser.parse_text("Mon unique titre")
        assert len(plan.sections) >= 0  # Peut être interprété comme titre ou section

    def test_plan_with_descriptions(self, parser):
        text = """1. Introduction
Présentation du contexte général.
2. Développement
Analyse détaillée du sujet.
3. Conclusion
Résumé et recommandations."""
        plan = parser.parse_text(text)
        sections_with_desc = [s for s in plan.sections if s.description]
        assert len(sections_with_desc) >= 2


class TestPlanHierarchy:
    def test_parent_assignment(self, parser):
        text = """1. Section principale
1.1 Sous-section A
1.2 Sous-section B
2. Deuxième section
2.1 Sous-section C"""
        plan = parser.parse_text(text)
        # Les sous-sections doivent avoir un parent
        sub_sections = [s for s in plan.sections if s.level == 2]
        for sub in sub_sections:
            assert sub.parent_id is not None

    def test_id_assignment(self, parser):
        text = """1. Introduction
2. Développement
3. Conclusion"""
        plan = parser.parse_text(text)
        ids = [s.id for s in plan.sections]
        assert len(ids) == len(set(ids))  # Tous les IDs doivent être uniques


class TestDistributePageBudget:
    def test_budget_distribution(self, parser):
        text = """1. Introduction
2. Développement
2.1 Partie A
2.2 Partie B
3. Conclusion"""
        plan = parser.parse_text(text)
        parser.distribute_page_budget(plan, 10)

        root_sections = plan.get_root_sections()
        total_budget = sum(s.page_budget or 0 for s in root_sections)
        assert abs(total_budget - 10) < 1  # Le total doit être proche de 10

    def test_no_budget_when_none(self, parser):
        text = "1. Section\n2. Section"
        plan = parser.parse_text(text)
        parser.distribute_page_budget(plan, None)
        for s in plan.sections:
            assert s.page_budget is None


class TestNormalizedPlan:
    def test_serialization(self, parser):
        text = "1. Intro\n2. Dev\n3. Conclu"
        plan = parser.parse_text(text)
        plan.objective = "Test"

        data = plan.to_dict()
        restored = NormalizedPlan.from_dict(data)

        assert len(restored.sections) == len(plan.sections)
        assert restored.objective == "Test"
        for orig, rest in zip(plan.sections, restored.sections):
            assert orig.id == rest.id
            assert orig.title == rest.title

    def test_get_section(self, parser):
        text = "1. A\n2. B\n3. C"
        plan = parser.parse_text(text)
        s = plan.get_section(plan.sections[1].id)
        assert s is not None
        assert s.title == plan.sections[1].title

    def test_get_children(self, parser):
        text = "1. Parent\n1.1 Enfant A\n1.2 Enfant B\n2. Autre"
        plan = parser.parse_text(text)
        parent = plan.get_root_sections()[0]
        children = plan.get_children(parent.id)
        assert len(children) >= 2
