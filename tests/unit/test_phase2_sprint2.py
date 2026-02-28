"""Tests unitaires pour la Phase 2 Sprint 2 — Robustesse & Scalabilité.

Couvre :
1. Contrôles mémoire Docling (file size + timeout)
2. Alternative LLM à GROBID (Structured Output JSON)
3. Support Batch API pour les Rédacteurs multi-agents
"""

import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.core.text_extractor import (
    ExtractionResult,
    _DoclingFileTooLarge,
    _DoclingTimeoutError,
    _extract_pdf_docling,
    _load_pdf_extraction_config,
    extract_pdf,
)
from src.core.orchestrator import (
    ProjectState,
    _extract_metadata_via_llm,
    _extract_first_pages,
)
from src.core.multi_agent_orchestrator import (
    GenerationResult,
    MultiAgentOrchestrator,
    build_dag,
    get_ready_sections,
)
from src.core.agent_framework import AgentConfig, AgentState
from src.providers.base import AIResponse, BatchRequest


# ═══════════════════════════════════════════════════════════════════
# 1. Contrôles mémoire Docling
# ═══════════════════════════════════════════════════════════════════


class TestDoclingConfigKeys:
    """Vérifie que les nouvelles clés de config sont chargées."""

    def test_config_includes_max_file_size(self):
        cfg = _load_pdf_extraction_config()
        assert "docling_max_file_size_mb" in cfg
        assert isinstance(cfg["docling_max_file_size_mb"], (int, float))
        assert cfg["docling_max_file_size_mb"] > 0

    def test_config_includes_timeout(self):
        cfg = _load_pdf_extraction_config()
        assert "docling_timeout_seconds" in cfg
        assert isinstance(cfg["docling_timeout_seconds"], (int, float))
        assert cfg["docling_timeout_seconds"] > 0


class TestDoclingFileSizeLimit:
    """Vérifie que les fichiers trop lourds contournent Docling → pymupdf."""

    @patch("src.core.text_extractor._load_pdf_extraction_config")
    @patch("src.core.text_extractor.os.path.getsize")
    def test_file_too_large_raises(self, mock_getsize, mock_config):
        """Un PDF de 60 Mo doit lever _DoclingFileTooLarge si la limite est 50 Mo."""
        mock_config.return_value = {
            "docling_page_batch_size": 30,
            "docling_batch_threshold": 50,
            "coverage_threshold": 0.80,
            "disable_page_images": True,
            "disable_picture_classification": True,
            "disable_ocr": True,
            "docling_max_file_size_mb": 50,
            "docling_timeout_seconds": 180,
        }
        mock_getsize.return_value = 60 * 1024 * 1024  # 60 Mo

        with pytest.raises(_DoclingFileTooLarge):
            _extract_pdf_docling(Path("/fake/large.pdf"))

    @patch("src.core.text_extractor._detect_pdf_libraries")
    @patch("src.core.text_extractor._PDF_EXTRACTORS", new_callable=dict)
    @patch("src.core.text_extractor._load_pdf_extraction_config")
    @patch("src.core.text_extractor.os.path.getsize")
    def test_file_too_large_falls_back_to_pymupdf(
        self, mock_getsize, mock_config, mock_extractors, mock_detect, tmp_path
    ):
        """Un fichier trop lourd doit être extrait via pymupdf (fallback)."""
        mock_config.return_value = {
            "docling_page_batch_size": 30,
            "docling_batch_threshold": 50,
            "coverage_threshold": 0.80,
            "disable_page_images": True,
            "disable_picture_classification": True,
            "disable_ocr": True,
            "docling_max_file_size_mb": 50,
            "docling_timeout_seconds": 180,
        }
        # 60 Mo pour _extract_pdf_docling, mais les autres calls use real os.path.getsize
        def getsize_side_effect(path):
            return 60 * 1024 * 1024  # toujours 60 Mo

        mock_getsize.side_effect = getsize_side_effect
        mock_detect.return_value = ["docling", "pymupdf"]
        mock_extractors["pymupdf"] = MagicMock(return_value=("Texte extrait par pymupdf", 5))

        f = tmp_path / "large.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")

        result = extract_pdf(f)
        assert result.status == "success"
        assert result.extraction_method == "pymupdf"
        assert "pymupdf" in result.text.lower() or result.text == "Texte extrait par pymupdf"


class TestDoclingTimeout:
    """Vérifie que le timeout Docling déclenche le fallback."""

    @patch("src.core.text_extractor._load_pdf_extraction_config")
    @patch("src.core.text_extractor.os.path.getsize")
    @patch("src.core.text_extractor._get_pdf_page_count")
    @patch("src.core.text_extractor.ProcessPoolExecutor")
    def test_timeout_raises_docling_timeout(
        self, mock_executor_cls, mock_page_count, mock_getsize, mock_config
    ):
        """Un future.result() qui dépasse le timeout lève _DoclingTimeoutError."""
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        mock_config.return_value = {
            "docling_page_batch_size": 30,
            "docling_batch_threshold": 50,
            "coverage_threshold": 0.80,
            "disable_page_images": True,
            "disable_picture_classification": True,
            "disable_ocr": True,
            "docling_max_file_size_mb": 50,
            "docling_timeout_seconds": 1,  # 1 seconde de timeout
        }
        mock_getsize.return_value = 10 * 1024 * 1024  # 10 Mo (sous la limite)
        mock_page_count.return_value = 10  # PDF modéré, single-pass

        # Simuler un timeout sur future.result()
        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()
        mock_future.cancel.return_value = True

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        with pytest.raises(_DoclingTimeoutError):
            _extract_pdf_docling(Path("/fake/slow.pdf"))

    @patch("src.core.text_extractor._detect_pdf_libraries")
    @patch("src.core.text_extractor._PDF_EXTRACTORS", new_callable=dict)
    @patch("src.core.text_extractor._load_pdf_extraction_config")
    @patch("src.core.text_extractor.os.path.getsize")
    @patch("src.core.text_extractor._get_pdf_page_count")
    @patch("src.core.text_extractor.ProcessPoolExecutor")
    def test_timeout_falls_back_to_pymupdf(
        self, mock_executor_cls, mock_page_count, mock_getsize,
        mock_config, mock_extractors, mock_detect, tmp_path
    ):
        """Après un timeout Docling, l'extraction passe à pymupdf."""
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        mock_config.return_value = {
            "docling_page_batch_size": 30,
            "docling_batch_threshold": 50,
            "coverage_threshold": 0.80,
            "disable_page_images": True,
            "disable_picture_classification": True,
            "disable_ocr": True,
            "docling_max_file_size_mb": 50,
            "docling_timeout_seconds": 1,
        }
        mock_getsize.return_value = 10 * 1024 * 1024
        mock_page_count.return_value = 10

        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()
        mock_future.cancel.return_value = True

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        mock_detect.return_value = ["docling", "pymupdf"]
        mock_extractors["pymupdf"] = MagicMock(return_value=("Texte après timeout", 10))

        f = tmp_path / "slow.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")

        result = extract_pdf(f)
        assert result.status == "success"
        assert result.extraction_method == "pymupdf"


# ═══════════════════════════════════════════════════════════════════
# 2. Alternative LLM à GROBID
# ═══════════════════════════════════════════════════════════════════


class MockProviderForMeta:
    """Provider mock qui retourne du JSON structuré."""

    def __init__(self, json_response: dict):
        self._json_response = json_response
        self.name = "mock"

    def generate_json(self, prompt, system_prompt=None, model=None,
                      temperature=0.3, max_tokens=512):
        return AIResponse(
            content=json.dumps(self._json_response),
            model=model or "mock-model",
            provider="mock",
            input_tokens=100,
            output_tokens=50,
        )

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        return AIResponse(
            content=json.dumps(self._json_response),
            model=model or "mock-model",
            provider="mock",
            input_tokens=100,
            output_tokens=50,
        )


class TestLLMMetadataExtraction:
    """Vérifie l'extraction de métadonnées via LLM (alternative GROBID)."""

    def _make_ext(self, filename="article.pdf", text=""):
        """Crée un ExtractionResult minimal."""
        return ExtractionResult(
            text=text,
            page_count=5,
            char_count=len(text),
            word_count=len(text.split()) if text else 0,
            extraction_method="pymupdf",
            status="success",
            source_filename=filename,
            source_size_bytes=1000,
            metadata={},
        )

    def test_full_metadata_extraction(self, tmp_path):
        """Le LLM retourne toutes les métadonnées correctement."""
        provider = MockProviderForMeta({
            "title": "Deep Learning for NLP",
            "authors": ["Smith, J.", "Doe, A."],
            "year": 2023,
            "doi": "10.1234/test.2023",
        })
        ext = self._make_ext(text="Deep Learning for NLP by Smith and Doe, 2023. " * 100)

        result = _extract_metadata_via_llm(ext, tmp_path, provider)

        assert result is not None
        assert result["title"] == "Deep Learning for NLP"
        assert result["authors"] == ["Smith, J.", "Doe, A."]
        assert result["year"] == 2023
        assert result["doi"] == "10.1234/test.2023"
        assert "author" in result  # Champ aplati

    def test_partial_metadata_with_empty_fields(self, tmp_path):
        """Le JSON est correctement parsé même si le LLM renvoie des champs vides/null."""
        provider = MockProviderForMeta({
            "title": "Some Paper Title",
            "authors": [],
            "year": None,
            "doi": None,
        })
        ext = self._make_ext(text="Some Paper Title. Introduction section..." * 50)

        result = _extract_metadata_via_llm(ext, tmp_path, provider)

        assert result is not None
        assert result["title"] == "Some Paper Title"
        assert "authors" not in result  # Empty list → not included
        assert "year" not in result     # None → not included
        assert "doi" not in result      # None → not included

    def test_empty_text_returns_none(self, tmp_path):
        """Un texte vide ne déclenche pas d'appel LLM."""
        provider = MockProviderForMeta({})
        ext = self._make_ext(text="")

        result = _extract_metadata_via_llm(ext, tmp_path, provider)
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        """Un JSON invalide retourné par le LLM retourne None."""
        provider = MagicMock()
        provider.name = "mock"
        provider.generate_json = MagicMock(return_value=AIResponse(
            content="Ceci n'est pas du JSON",
            model="mock",
            provider="mock",
        ))
        ext = self._make_ext(text="Some text content " * 100)

        result = _extract_metadata_via_llm(ext, tmp_path, provider)
        assert result is None

    def test_provider_without_generate_json_uses_generate(self, tmp_path):
        """Si le provider n'a pas generate_json, utilise generate classique."""
        provider = MagicMock(spec=[])
        provider.name = "mock"
        provider.generate = MagicMock(return_value=AIResponse(
            content=json.dumps({
                "title": "Fallback Title",
                "authors": ["Author A"],
                "year": 2022,
                "doi": None,
            }),
            model="mock",
            provider="mock",
        ))
        # S'assurer que generate_json n'existe PAS
        assert not hasattr(provider, "generate_json")

        ext = self._make_ext(text="Fallback Title by Author A, 2022." * 50)
        result = _extract_metadata_via_llm(ext, tmp_path, provider)

        assert result is not None
        assert result["title"] == "Fallback Title"
        provider.generate.assert_called_once()

    def test_authors_as_string(self, tmp_path):
        """Si le LLM retourne authors comme une chaîne, la normalisation fonctionne."""
        provider = MockProviderForMeta({
            "title": "Paper Title",
            "authors": "Single Author",
            "year": 2021,
            "doi": None,
        })
        ext = self._make_ext(text="Paper Title by Single Author, 2021." * 50)

        result = _extract_metadata_via_llm(ext, tmp_path, provider)

        assert result is not None
        assert result["authors"] == ["Single Author"]
        assert result["author"] == "Single Author"


class TestExtractFirstPages:
    """Vérifie l'extraction des premières pages d'un PDF."""

    def test_fallback_to_extraction_text(self, tmp_path):
        """Si pymupdf n'est pas dispo, utilise le texte déjà extrait."""
        ext = MagicMock()
        ext.source_filename = "test.pdf"
        ext.text = "Page 1 content. " * 1000

        result = _extract_first_pages(ext, tmp_path, max_pages=2)

        assert result is not None
        assert len(result) <= 2 * 3000
        assert result.startswith("Page 1 content.")


# ═══════════════════════════════════════════════════════════════════
# 3. Support Batch API pour les Rédacteurs
# ═══════════════════════════════════════════════════════════════════


class MockProviderBatch:
    """Provider mock avec support batch."""

    def __init__(self):
        self.name = "mock"
        self.submitted_batches = []

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        return AIResponse(
            content="Generated content",
            model=model or "mock-model",
            provider="mock",
            input_tokens=100,
            output_tokens=50,
        )

    def is_available(self):
        return True

    def get_default_model(self):
        return "mock-model"

    def list_models(self):
        return ["mock-model"]

    def supports_batch(self):
        return True

    def submit_batch(self, requests):
        self.submitted_batches.append(requests)
        return f"batch_{len(self.submitted_batches)}"

    def poll_batch(self, batch_id):
        from src.providers.base import BatchStatus, BatchStatusEnum
        return BatchStatus(
            batch_id=batch_id,
            status=BatchStatusEnum.COMPLETED,
            total=5,
            completed=5,
        )

    def retrieve_batch_results(self, batch_id):
        if not self.submitted_batches:
            return {}
        last_batch = self.submitted_batches[-1]
        return {req.custom_id: f"Content for {req.custom_id}" for req in last_batch}


class MockProviderNoBatch:
    """Provider mock SANS support batch."""

    def __init__(self):
        self.name = "mock_no_batch"

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096):
        return AIResponse(
            content="Generated content (realtime)",
            model=model or "mock-model",
            provider="mock_no_batch",
            input_tokens=100,
            output_tokens=50,
        )

    def is_available(self):
        return True

    def get_default_model(self):
        return "mock-model"

    def list_models(self):
        return ["mock-model"]

    def supports_batch(self):
        return False


def _make_state_for_batch(n_sections=5, use_batch_api=True):
    """Crée un ProjectState configuré pour le mode batch."""
    from src.core.plan_parser import NormalizedPlan, PlanSection

    sections = [
        PlanSection(id=f"s{i:02d}", title=f"Section {i}", level=1, description=f"Desc {i}")
        for i in range(1, n_sections + 1)
    ]
    plan = NormalizedPlan(sections=sections, title="Doc test", objective="Test batch")

    state = ProjectState(name="test_batch")
    state.plan = plan
    state.config = {
        "multi_agent": {
            "enabled": True,
            "use_batch_api": use_batch_api,
            "max_parallel_writers": 4,
            "max_parallel_verifiers": 4,
            "quality_threshold": 3.5,
            "section_correction_threshold": 3.0,
            "max_correction_passes": 1,
            "max_cost_usd": 10.0,
            "agents": {
                "architecte": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "redacteur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "verificateur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "evaluateur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
                "correcteur": {"provider": "mock", "model": "mock-model", "timeout_s": 30},
            },
        },
    }
    return state


class TestAgentConfigBatchFlag:
    """Vérifie que use_batch_api est correctement parsé."""

    def test_batch_api_default_false(self):
        config = AgentConfig.from_config({})
        assert config.use_batch_api is False

    def test_batch_api_true(self):
        config = AgentConfig.from_config({
            "multi_agent": {"use_batch_api": True},
        })
        assert config.use_batch_api is True


class TestBatchModeOrchestrator:
    """Vérifie le mode batch du multi-agent orchestrator."""

    def test_batch_mode_submits_single_batch(self):
        """5 sections parallèles → un seul appel submit_batch (pas 5 appels)."""
        state = _make_state_for_batch(5, use_batch_api=True)
        provider = MockProviderBatch()
        config = AgentConfig.from_config(state.config)
        providers = {"mock": provider}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        # Simuler un DAG plat (toutes les sections indépendantes)
        architecture = {
            "sections": [
                {"id": f"s{i:02d}", "title": f"Section {i}"}
                for i in range(1, 6)
            ],
            "dependances": {f"s{i:02d}": [] for i in range(1, 6)},
            "system_prompt_global": "Tu es un rédacteur.",
        }
        orch._dag = build_dag(architecture["dependances"])

        generated = asyncio.run(orch._run_generation_phase(architecture))

        # Vérifier : un seul batch soumis contenant les 5 sections
        assert len(provider.submitted_batches) == 1
        batch = provider.submitted_batches[0]
        assert len(batch) == 5
        batch_ids = {req.custom_id for req in batch}
        assert batch_ids == {"s01", "s02", "s03", "s04", "s05"}

    def test_batch_mode_sets_waiting_status(self):
        """Le statut passe à 'waiting_for_batch' après soumission."""
        state = _make_state_for_batch(3, use_batch_api=True)
        provider = MockProviderBatch()
        config = AgentConfig.from_config(state.config)
        providers = {"mock": provider}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        architecture = {
            "sections": [{"id": f"s{i:02d}", "title": f"Section {i}"} for i in range(1, 4)],
            "dependances": {f"s{i:02d}": [] for i in range(1, 4)},
            "system_prompt_global": "",
        }
        orch._dag = build_dag(architecture["dependances"])

        asyncio.run(orch._run_generation_phase(architecture))

        assert state.current_step == "waiting_for_batch"
        assert state.batch_id is not None
        assert state.batch_id.startswith("batch_")

    def test_batch_mode_stores_batch_id(self):
        """Le batch_id est enregistré dans state pour la reprise."""
        state = _make_state_for_batch(2, use_batch_api=True)
        provider = MockProviderBatch()
        config = AgentConfig.from_config(state.config)
        providers = {"mock": provider}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        architecture = {
            "sections": [{"id": "s01", "title": "S1"}, {"id": "s02", "title": "S2"}],
            "dependances": {"s01": [], "s02": []},
            "system_prompt_global": "",
        }
        orch._dag = build_dag(architecture["dependances"])

        asyncio.run(orch._run_generation_phase(architecture))

        assert state.batch_id is not None
        assert state.batch_architecture == architecture

    def test_batch_resume_continues_dag(self):
        """La reprise après batch débloque les sections descendantes du DAG."""
        state = _make_state_for_batch(4, use_batch_api=True)
        provider = MockProviderBatch()
        config = AgentConfig.from_config(state.config)
        providers = {"mock": provider}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        # DAG : s01, s02 indépendants → s03 dépend de s01 → s04 dépend de s02
        architecture = {
            "sections": [
                {"id": "s01", "title": "S1"},
                {"id": "s02", "title": "S2"},
                {"id": "s03", "title": "S3"},
                {"id": "s04", "title": "S4"},
            ],
            "dependances": {
                "s01": [],
                "s02": [],
                "s03": ["s01"],
                "s04": ["s02"],
            },
            "system_prompt_global": "",
        }
        orch._dag = build_dag(architecture["dependances"])

        # Premier lancement : soumet s01 et s02
        generated = asyncio.run(orch._run_generation_phase(architecture))
        assert len(provider.submitted_batches) == 1
        first_batch_ids = {req.custom_id for req in provider.submitted_batches[0]}
        assert first_batch_ids == {"s01", "s02"}

        # Simuler les résultats du batch
        batch_results = {"s01": "Content s01", "s02": "Content s02"}

        # Reprendre → soumet s03 et s04
        generated = asyncio.run(
            orch.resume_from_batch(batch_results, architecture)
        )

        # Vérifier que 2 batches ont été soumis au total
        assert len(provider.submitted_batches) == 2
        second_batch_ids = {req.custom_id for req in provider.submitted_batches[1]}
        assert second_batch_ids == {"s03", "s04"}


class TestBatchRequestConstruction:
    """Vérifie la construction des BatchRequest."""

    def test_build_writer_prompt(self):
        """Le prompt de rédaction batch contient les infos de la section."""
        state = _make_state_for_batch(1, use_batch_api=True)
        provider = MockProviderBatch()
        config = AgentConfig.from_config(state.config)
        providers = {"mock": provider}

        orch = MultiAgentOrchestrator(
            project_state=state,
            agent_config=config,
            providers=providers,
        )

        prompt = orch._build_writer_prompt(
            sid="s01",
            section_info={"title": "Introduction", "type": "fond", "ton": "formel", "longueur_cible": 800},
            corpus_text="Texte du corpus...",
            system_prompt_global="System prompt",
            prereqs={"s00": "Résumé prérequis"},
        )

        assert "Introduction" in prompt
        assert "s01" in prompt
        assert "Texte du corpus" in prompt
        assert "Résumé prérequis" in prompt


class TestProjectStateBatchFields:
    """Vérifie la sérialisation des champs batch dans ProjectState."""

    def test_batch_fields_in_to_dict(self):
        state = ProjectState(name="test")
        state.batch_id = "batch_123"
        state.batch_generated = {"s01": "content"}
        state.batch_architecture = {"sections": []}

        d = state.to_dict()
        assert d["batch_id"] == "batch_123"
        assert d["batch_generated"] == {"s01": "content"}
        assert d["batch_architecture"] == {"sections": []}

    def test_batch_fields_from_dict(self):
        data = {
            "name": "test",
            "batch_id": "batch_456",
            "batch_generated": {"s01": "content", "s02": "content2"},
            "batch_architecture": {"sections": [], "dependances": {}},
        }
        state = ProjectState.from_dict(data)
        assert state.batch_id == "batch_456"
        assert state.batch_generated == {"s01": "content", "s02": "content2"}
        assert state.batch_architecture == {"sections": [], "dependances": {}}

    def test_batch_fields_default_none(self):
        state = ProjectState.from_dict({"name": "test"})
        assert state.batch_id is None
        assert state.batch_generated == {}
        assert state.batch_architecture is None
