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

    def test_with_full_excerpts_digest(self, engine):
        digest = {
            "tier": "full_excerpts",
            "num_documents": 2,
            "entries": [
                {"source_file": "rapport.pdf", "text": "Résumé du rapport annuel 2024."},
                {"source_file": "étude.docx", "text": "Analyse du marché européen."},
            ],
        }
        prompt = engine.build_plan_generation_prompt(
            "Synthèse stratégique", 15, corpus_digest=digest,
        )
        assert "rapport.pdf" in prompt
        assert "étude.docx" in prompt
        assert "Résumé du rapport annuel" in prompt
        assert "Analyse du marché européen" in prompt
        assert "2 document(s)" in prompt
        assert "Tiens compte du contenu" in prompt

    def test_with_first_sentences_digest(self, engine):
        digest = {
            "tier": "first_sentences",
            "num_documents": 25,
            "entries": [
                {"source_file": f"doc_{i}.pdf", "text": f"Phrase du document {i}."}
                for i in range(25)
            ],
        }
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "25 documents" in prompt
        assert "doc_0.pdf" in prompt
        assert "doc_24.pdf" in prompt
        assert "Phrase du document 0" in prompt

    def test_with_sampled_digest(self, engine):
        digest = {
            "tier": "sampled",
            "num_documents": 100,
            "entries": [
                {"source_file": "doc_000.pdf", "text": "Extrait du doc 0."},
                {"source_file": "doc_050.pdf", "text": "Extrait du doc 50."},
            ],
            "all_filenames": [f"doc_{i:03d}.pdf" for i in range(100)],
        }
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "100 documents" in prompt
        assert "Liste des sources" in prompt
        assert "doc_000.pdf" in prompt
        assert "doc_099.pdf" in prompt  # in filenames list
        assert "Extrait du doc 0" in prompt
        assert "échantillon de 2 documents" in prompt

    def test_without_corpus_digest(self, engine):
        prompt = engine.build_plan_generation_prompt("Objectif test", 10, corpus_digest=None)
        assert "corpus" not in prompt.lower()
        assert "Tiens compte" not in prompt

    def test_with_empty_digest(self, engine):
        digest = {"tier": "full_excerpts", "num_documents": 0, "entries": []}
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "corpus" not in prompt.lower()
        assert "Tiens compte" not in prompt


class TestBuildSummaryPrompt:
    def test_summary_prompt(self, engine):
        prompt = engine.build_summary_prompt("Introduction", "Le contenu de la section introduction.")
        assert "Introduction" in prompt
        assert "contenu de la section" in prompt
