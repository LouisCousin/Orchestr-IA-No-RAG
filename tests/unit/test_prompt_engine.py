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
        assert "aucun corpus" in prompt.lower()
        assert "NEEDS_SOURCE" in prompt

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
                {
                    "source_file": f"doc_{i}.pdf",
                    "text": f"Phrase du document {i}.",
                    "keywords": ["cloud", "infrastructure"],
                }
                for i in range(25)
            ],
        }
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "25 documents" in prompt
        assert "doc_0.pdf" in prompt
        assert "doc_24.pdf" in prompt
        assert "Phrase du document 0" in prompt
        assert "cloud, infrastructure" in prompt

    def test_first_sentences_without_keywords(self, engine):
        digest = {
            "tier": "first_sentences",
            "num_documents": 12,
            "entries": [
                {"source_file": f"doc_{i}.pdf", "text": f"Phrase {i}."}
                for i in range(12)
            ],
        }
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "12 documents" in prompt
        assert "doc_0.pdf" in prompt

    def test_with_sampled_digest_with_keywords(self, engine):
        digest = {
            "tier": "sampled",
            "num_documents": 100,
            "entries": [
                {"source_file": "doc_000.pdf", "text": "Extrait du doc 0."},
                {"source_file": "doc_050.pdf", "text": "Extrait du doc 50."},
            ],
            "all_filenames": [f"doc_{i:03d}.pdf" for i in range(100)],
            "all_files_keywords": [
                {"source_file": f"doc_{i:03d}.pdf", "keywords": ["marché", "finance"]}
                for i in range(100)
            ],
        }
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "100 documents" in prompt
        assert "thématiques" in prompt.lower()
        assert "doc_000.pdf" in prompt
        assert "doc_099.pdf" in prompt
        assert "marché, finance" in prompt
        assert "Extrait du doc 0" in prompt
        assert "échantillon de 2 documents" in prompt

    def test_with_sampled_digest_without_keywords(self, engine):
        digest = {
            "tier": "sampled",
            "num_documents": 100,
            "entries": [
                {"source_file": "doc_000.pdf", "text": "Extrait du doc 0."},
            ],
            "all_filenames": [f"doc_{i:03d}.pdf" for i in range(100)],
        }
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "100 documents" in prompt
        assert "doc_099.pdf" in prompt

    def test_without_corpus_digest(self, engine):
        prompt = engine.build_plan_generation_prompt("Objectif test", 10, corpus_digest=None)
        assert "corpus" not in prompt.lower()
        assert "Tiens compte" not in prompt

    def test_with_empty_digest(self, engine):
        digest = {"tier": "full_excerpts", "num_documents": 0, "entries": []}
        prompt = engine.build_plan_generation_prompt("Objectif", 10, corpus_digest=digest)
        assert "corpus" not in prompt.lower()
        assert "Tiens compte" not in prompt


class TestBuildSystemPromptHasCorpus:
    def test_system_prompt_with_corpus(self, engine):
        """Anti-hallucination inclus quand has_corpus=True."""
        prompt = engine.build_system_prompt(has_corpus=True)
        assert "RÈGLES DE FIABILITÉ" in prompt

    def test_system_prompt_without_corpus(self, engine):
        """Anti-hallucination exclu quand has_corpus=False (CA3-4)."""
        prompt = engine.build_system_prompt(has_corpus=False)
        assert "RÈGLES DE FIABILITÉ" not in prompt


class TestNoMarkdownInTemplates:
    """CA2-3 : Les templates de prompt ne contiennent plus aucun ##."""

    def test_section_template_no_markdown_headers(self, engine, sample_plan, sample_chunks):
        section = sample_plan.sections[0]
        prompt = engine.build_section_prompt(
            section=section, plan=sample_plan,
            corpus_chunks=sample_chunks, previous_summaries=[],
        )
        # Le prompt ne doit pas contenir de ## (en-tête markdown)
        for line in prompt.split("\n"):
            stripped = line.strip()
            assert not stripped.startswith("##"), f"Found markdown header: {stripped}"

    def test_refinement_template_no_markdown_headers(self, engine, sample_plan, sample_chunks):
        section = sample_plan.sections[0]
        prompt = engine.build_refinement_prompt(
            section=section, plan=sample_plan,
            draft_content="Brouillon test",
            corpus_chunks=sample_chunks, previous_summaries=[],
        )
        for line in prompt.split("\n"):
            stripped = line.strip()
            assert not stripped.startswith("##"), f"Found markdown header: {stripped}"

    def test_plan_template_no_markdown_headers(self, engine):
        prompt = engine.build_plan_generation_prompt("Objectif test", 10)
        for line in prompt.split("\n"):
            stripped = line.strip()
            assert not stripped.startswith("##"), f"Found markdown header: {stripped}"


class TestCorpusChunksGroupedFormats:
    """CA3-6, CA3-7 : Regroupement des chunks (objets et dicts)."""

    def test_grouped_by_source_objects(self, engine):
        """CA3-6 : chunks sous forme d'objets regroupés par source."""
        chunks = [
            CorpusChunk(text="Extrait 1 du doc A", source_file="docA.pdf", chunk_index=0),
            CorpusChunk(text="Extrait 2 du doc A", source_file="docA.pdf", chunk_index=1),
            CorpusChunk(text="Extrait 1 du doc B", source_file="docB.pdf", chunk_index=0),
        ]
        result = PromptEngine._format_corpus_chunks_grouped(chunks)
        # Deux groupes
        assert "[Document : docA.pdf]" in result
        assert "[Document : docB.pdf]" in result
        assert "Extrait 1 du doc A" in result
        assert "Extrait 2 du doc A" in result

    def test_grouped_by_source_dicts(self, engine):
        """CA3-7 : chunks sous forme de dicts."""
        chunks = [
            {"text": "Texte chunk 1", "source_file": "rapport.pdf"},
            {"text": "Texte chunk 2", "source_file": "rapport.pdf"},
            {"text": "Texte chunk 3", "source_file": "etude.pdf"},
        ]
        result = PromptEngine._format_corpus_chunks_grouped(chunks)
        assert "[Document : rapport.pdf]" in result
        assert "[Document : etude.pdf]" in result

    def test_grouped_mixed_formats(self, engine):
        """CA3-7 : chunks hétérogènes (mélange objets et dicts)."""
        chunks = [
            CorpusChunk(text="Objet chunk", source_file="doc1.pdf", chunk_index=0),
            {"text": "Dict chunk", "source_file": "doc2.pdf"},
        ]
        result = PromptEngine._format_corpus_chunks_grouped(chunks)
        assert "[Document : doc1.pdf]" in result
        assert "[Document : doc2.pdf]" in result

    def test_no_source_numbered_references(self, engine, sample_plan, sample_chunks):
        """Le prompt ne doit contenir aucun [Source N]."""
        section = sample_plan.sections[0]
        prompt = engine.build_section_prompt(
            section=section, plan=sample_plan,
            corpus_chunks=sample_chunks, previous_summaries=[],
        )
        import re
        matches = re.findall(r'\[Source\s*\d+\]', prompt)
        assert len(matches) == 0, f"Found [Source N] references: {matches}"


class TestCorpusEmptyFallback:
    """CA3-4 : Sans corpus, le prompt ne contient pas de contradiction."""

    def test_no_corpus_mentions_needs_source(self, engine, sample_plan):
        section = sample_plan.sections[0]
        prompt = engine.build_section_prompt(
            section=section, plan=sample_plan,
            corpus_chunks=[], previous_summaries=[],
        )
        assert "NEEDS_SOURCE" in prompt
        assert "aucun corpus" in prompt.lower()


class TestBuildSummaryPrompt:
    def test_summary_prompt(self, engine):
        prompt = engine.build_summary_prompt("Introduction", "Le contenu de la section introduction.")
        assert "Introduction" in prompt
        assert "contenu de la section" in prompt
