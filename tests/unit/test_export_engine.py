"""Tests unitaires pour le module export_engine."""

import pytest
from pathlib import Path

from src.core.export_engine import ExportEngine, hex_to_rgb
from src.core.plan_parser import NormalizedPlan, PlanSection


@pytest.fixture
def sample_plan():
    plan = NormalizedPlan(title="Rapport de test", objective="Tester l'export DOCX")
    plan.sections = [
        PlanSection(id="1", title="Introduction", level=1),
        PlanSection(id="1.1", title="Contexte", level=2, parent_id="1"),
        PlanSection(id="2", title="Analyse", level=1),
        PlanSection(id="3", title="Conclusion", level=1),
    ]
    return plan


@pytest.fixture
def generated_sections():
    return {
        "1": "L'introduction présente le contexte général du rapport.",
        "1.1": "Le contexte détaillé de notre analyse.\n\n- Point un\n- Point deux\n- Point trois",
        "2": "L'analyse approfondie des données recueillies.\n\n**Résultats clés**\n\nLes résultats montrent une tendance positive.",
        "3": "En conclusion, les résultats sont prometteurs.",
    }


@pytest.fixture
def engine():
    return ExportEngine()


class TestHexToRgb:
    def test_valid_hex(self):
        color = hex_to_rgb("#F0C441")
        assert color[0] == 240
        assert color[1] == 196
        assert color[2] == 65

    def test_hex_without_hash(self):
        color = hex_to_rgb("4E4E50")
        assert color[0] == 78
        assert color[1] == 78
        assert color[2] == 80


class TestExportDocx:
    def test_export_creates_file(self, engine, sample_plan, generated_sections, tmp_path):
        output = tmp_path / "output.docx"
        result = engine.export_docx(sample_plan, generated_sections, output, "Test")
        assert result.exists()
        assert result.suffix == ".docx"
        assert result.stat().st_size > 0

    def test_export_with_custom_styling(self, sample_plan, generated_sections, tmp_path):
        custom_engine = ExportEngine(styling={
            "primary_color": "#FF0000",
            "secondary_color": "#0000FF",
            "font_title": "Arial",
            "font_body": "Times New Roman",
            "font_size_title": 20,
            "font_size_body": 12,
        })
        output = tmp_path / "custom_output.docx"
        result = custom_engine.export_docx(sample_plan, generated_sections, output, "Custom")
        assert result.exists()

    def test_export_with_missing_sections(self, engine, sample_plan, tmp_path):
        # Seulement 2 sections sur 4 générées
        partial_sections = {"1": "Introduction", "3": "Conclusion"}
        output = tmp_path / "partial.docx"
        result = engine.export_docx(sample_plan, partial_sections, output, "Partiel")
        assert result.exists()

    def test_export_empty_sections(self, engine, sample_plan, tmp_path):
        output = tmp_path / "empty.docx"
        result = engine.export_docx(sample_plan, {}, output, "Vide")
        assert result.exists()

    def test_export_creates_parent_dirs(self, engine, sample_plan, generated_sections, tmp_path):
        output = tmp_path / "sub" / "dir" / "output.docx"
        result = engine.export_docx(sample_plan, generated_sections, output, "Test")
        assert result.exists()


class TestExportMetadataExcel:
    def test_export_excel(self, engine, sample_plan, generated_sections, tmp_path):
        output = tmp_path / "metadata.xlsx"
        cost_report = {
            "total_input_tokens": 5000,
            "total_output_tokens": 2000,
            "total_cost_usd": 0.05,
            "entries": [
                {"section_id": "1", "model": "gpt-4o", "input_tokens": 1000,
                 "output_tokens": 500, "cost_usd": 0.01, "task_type": "generation"},
            ],
        }
        result = engine.export_metadata_excel(sample_plan, generated_sections, cost_report, output)
        assert result.exists()
        assert result.suffix == ".xlsx"
