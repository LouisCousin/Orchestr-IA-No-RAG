"""Tests unitaires pour le module prompt_engine."""

import pytest

from src.core.prompt_engine import PromptEngine
from src.core.plan_parser import PlanSection, NormalizedPlan
from src.core.corpus_extractor import CorpusChunk


@pytest.fixture
def engine():
    return PromptEngine()


@pytest.fixture
def engine_with_instructions():
    return PromptEngine(persistent_instructions="Utilise un ton formel et académique.")


@pytest.fixture
def sample_plan():
    plan = NormalizedPlan(title="Test", objective="Créer un rapport")
    plan.sections = [
        PlanSection(id="1", title="Introduction", level=1),
        PlanSection(id="2", title="Développement", level=1),
    ]
    return plan


@pytest.fixture
def sample_chunks():
    return [
        CorpusChunk(text="Bloc de corpus numéro 1.", source_file="doc1.txt", chunk_index=0),
        CorpusChunk(text="Bloc de corpus numéro 2.", source_file="doc2.txt", chunk_index=0),
    ]


class TestBuildSystemPrompt:
    def test_basic_system_prompt(self, engine):
        prompt = engine.build_system_prompt()
        assert "rédacteur professionnel" in prompt.lower()
        assert "français" in prompt.lower()

    def test_with_persistent_instructions(self, engine_with_instructions):
        prompt = engine_with_instructions.build_system_prompt()
        assert "formel et académique" in prompt


class TestBuildSectionPrompt:
    def test_basic_prompt(self, engine, sample_plan, sample_chunks):
        section = sample_plan.sections[0]
        prompt = engine.build_section_prompt(
            section=section,
            plan=sample_plan,
            corpus_chunks=sample_chunks,
            previous_summaries=[],
        )
        assert "Introduction" in prompt
        assert "Bloc de corpus numéro 1" in prompt
        assert "doc1.txt" in prompt

    def test_with_previous_context(self, engine, sample_plan, sample_chunks):
        section = sample_plan.sections[1]
        prompt = engine.build_section_prompt(
            section=section,
            plan=sample_plan,
            corpus_chunks=sample_chunks,
            previous_summaries=["Introduction : présentation du contexte."],
        )
        assert "Introduction" in prompt
        assert "présentation du contexte" in prompt

    def test_no_corpus(self, engine, sample_plan):
        section = sample_plan.sections[0]
        prompt = engine.build_section_prompt(
            section=section,
            plan=sample_plan,
            corpus_chunks=[],
            previous_summaries=[],
        )
        assert "connaissances générales" in prompt.lower()

    def test_with_page_budget(self, engine, sample_plan):
        section = sample_plan.sections[0]
        section.page_budget = 3.0
        prompt = engine.build_section_prompt(
            section=section,
            plan=sample_plan,
            corpus_chunks=[],
            previous_summaries=[],
        )
        assert "3.0" in prompt or "page" in prompt.lower()


class TestBuildPlanGenerationPrompt:
    def test_plan_prompt(self, engine):
        prompt = engine.build_plan_generation_prompt("Analyser le marché de l'IA", 20)
        assert "marché de l'IA" in prompt
        assert "20" in prompt

    def test_default_pages(self, engine):
        prompt = engine.build_plan_generation_prompt("Objectif test")
        assert "10" in prompt  # Valeur par défaut


class TestBuildSummaryPrompt:
    def test_summary_prompt(self, engine):
        prompt = engine.build_summary_prompt("Introduction", "Le contenu de la section introduction.")
        assert "Introduction" in prompt
        assert "contenu de la section" in prompt
