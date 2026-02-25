"""Tests d'intégration Phase 3 complète — interactions entre modules."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.persistent_instructions import PersistentInstructions
from src.core.glossary_engine import GlossaryEngine
from src.core.persona_engine import PersonaEngine
from src.core.template_library import TemplateLibrary
from src.core.feedback_engine import FeedbackEngine
from src.core.hitl_journal import HITLJournal
from src.core.metadata_overrides import MetadataOverrides
from src.core.citation_engine import CitationEngine
from src.core.profile_manager import ProfileManager


class TestPersistentInstructionsWithGlossary:
    """Intégration instructions persistantes + glossaire."""

    def test_instructions_and_glossary_for_section(self, tmp_path):
        """Instructions et glossaire s'appliquent à une même section."""
        pi = PersistentInstructions(project_dir=tmp_path)
        pi.set_project_instruction("style_ton", "Adopte un ton académique.")
        pi.set_section_instruction("sec_1", "contenu", "Cite les sources APA.")

        glossary = GlossaryEngine(project_dir=tmp_path, max_terms_per_prompt=15, enabled=True)
        glossary.add_term(term="IA", definition="Intelligence artificielle", domain="technologie")
        glossary.add_term(term="RAG", definition="Retrieval Augmented Generation", domain="IA")

        # Both should produce content for sec_1
        instructions = pi.get_instructions("sec_1")
        assert "ton académique" in instructions
        assert "sources APA" in instructions

        terms = glossary.get_terms_for_section("Introduction à l'IA")
        prompt_block = glossary.format_for_prompt(terms)
        assert "IA" in prompt_block
        assert "GLOSSAIRE" in prompt_block


class TestPersonaWithInstructions:
    """Intégration persona + instructions persistantes."""

    def test_persona_and_instructions_complement(self, tmp_path):
        """Persona et instructions persistent coexistent."""
        persona_engine = PersonaEngine(project_dir=tmp_path, enabled=True)
        pid = persona_engine.create(
            name="Gestionnaire",
            profile="Gestionnaire de projet senior",
            expertise_level="expert",
            register="formel",
        )
        persona_engine.assign_to_section("sec_1", pid)

        pi = PersistentInstructions(project_dir=tmp_path)
        pi.set_project_instruction("domaine_metier", "Domaine : gestion de projet informatique.")

        # Both produce relevant prompt content
        persona = persona_engine.get_persona_for_section("sec_1")
        persona_text = persona_engine.format_for_prompt(persona)
        assert "Gestionnaire de projet senior" in persona_text
        assert "expert" in persona_text

        instructions = pi.get_instructions("sec_1")
        assert "gestion de projet informatique" in instructions


class TestTemplateWithGlossary:
    """Intégration template library + glossaire."""

    def test_template_resolve_with_glossary_terms(self, tmp_path):
        """Les templates peuvent utiliser des termes du glossaire."""
        lib_path = tmp_path / "lib.json"
        lib_path.write_text("[]", encoding="utf-8")
        lib = TemplateLibrary(library_path=lib_path)

        tid = lib.create(
            name="Section technique",
            content="Rédige une section sur {sujet} en utilisant la terminologie correcte.",
            variables=[{"name": "sujet", "description": "Sujet"}],
        )

        glossary = GlossaryEngine(project_dir=tmp_path, enabled=True)
        glossary.add_term(term="Machine Learning", definition="Apprentissage automatique")

        # Resolve template
        resolved = lib.resolve(tid, {"sujet": "le Machine Learning"})
        assert "Machine Learning" in resolved

        # Glossary provides complementary info
        terms = glossary.get_terms_for_section("Machine Learning")
        assert len(terms) > 0


class TestMetadataOverrideWithCitations:
    """Intégration metadata overrides + citations."""

    def test_override_improves_citation(self, tmp_path):
        """Les overrides corrigent les métadonnées avant la citation APA."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        overrides = MetadataOverrides(project_dir=project_dir)

        # Simulate incomplete PDF metadata
        pdf_data = {"title": "Mon Article", "year": 2024}
        grobid_data = {"authors": ["Dupont, J."], "journal": "Nature"}

        # User adds overrides
        overrides.save_override("doc_001", {
            "volume": "42",
            "issue": "3",
            "pages_range": "100-120",
            "doi": "10.1234/test.001",
        })

        merged = overrides.merge_metadata("doc_001", grobid_data=grobid_data, pdf_data=pdf_data)
        assert merged["title"] == "Mon Article"
        assert merged["authors"] == ["Dupont, J."]
        assert merged["volume"] == "42"
        assert merged["doi"] == "10.1234/test.001"

        # Use merged data for APA citation
        ref = CitationEngine.format_apa_reference(
            doc_type="article",
            authors="Dupont, J.",
            year=merged["year"],
            title=merged["title"],
            journal=merged.get("journal"),
            volume=merged.get("volume"),
            issue=merged.get("issue"),
            pages_range=merged.get("pages_range"),
            doi=merged.get("doi"),
        )
        assert "Dupont" in ref
        assert "(2024)" in ref
        assert "10.1234/test.001" in ref


class TestHITLJournalWithCheckpoint:
    """Intégration HITL journal + checkpoint manager."""

    def test_journal_records_checkpoint_resolution(self, tmp_path):
        """Le journal HITL enregistre les résolutions de checkpoints."""
        journal_path = tmp_path / "hitl_journal.json"
        journal = HITLJournal(journal_path=journal_path)

        journal.log_intervention(
            project_name="test_project",
            checkpoint_type="after_plan_validation",
            intervention_type="modify",
            section_id=None,
            original_content="Plan v1",
            modified_content="Plan v2",
            delta_summary="Added section 3.2",
        )

        journal.log_intervention(
            project_name="test_project",
            checkpoint_type="after_generation",
            intervention_type="accept",
            section_id="sec_1.1",
        )

        stats = journal.get_statistics()
        assert stats["total"] == 2
        assert stats["by_type"]["modify"] == 1
        assert stats["by_type"]["accept"] == 1

        interventions = journal.get_interventions(project="test_project")
        assert len(interventions) == 2


class TestFeedbackWithProfile:
    """Intégration feedback + profils personnalisés."""

    def test_feedback_adjustments_persist_across_sections(self):
        """Les ajustements feedback sont disponibles pour les sections suivantes."""
        engine = FeedbackEngine(provider=None, enabled=True)

        # Simulate accepted feedback
        from src.core.feedback_engine import FeedbackEntry
        entry = FeedbackEntry(
            section_id="sec_1",
            category="style",
            suggestion="Use bullet points for key findings",
            decision="accepted",
        )
        engine._history.append(entry)

        adjustments = engine.get_active_adjustments()
        assert len(adjustments) == 1
        assert "bullet points" in adjustments[0]

        # These adjustments would be injected into subsequent prompts


class TestProfileManagerCustom:
    """Intégration profile manager avec profils personnalisés."""

    @patch("src.core.profile_manager.ROOT_DIR")
    def test_custom_profile_lifecycle(self, mock_root, tmp_path):
        mock_root.__truediv__ = lambda self, x: tmp_path / x
        mock_root.return_value = tmp_path

        pm = ProfileManager()
        pm.profiles_dir = tmp_path / "profiles" / "default"
        pm.custom_dir = tmp_path / "profiles" / "custom"
        pm.profiles_dir.mkdir(parents=True, exist_ok=True)
        pm.custom_dir.mkdir(parents=True, exist_ok=True)

        # Save custom profile
        path = pm.save_custom_profile(
            name="Mon profil custom",
            description="Un profil sur mesure",
            config={
                "target_pages": 30,
                "tone": "académique",
                "generation": {"temperature": 0.5},
            },
        )
        assert path.exists()

        # List custom profiles
        custom = pm.list_custom_profiles()
        assert len(custom) >= 1
        assert any(p["name"] == "Mon profil custom" for p in custom)

        # Delete custom profile
        profile_id = custom[0]["id"]
        assert pm.delete_custom_profile(profile_id) is True
        assert pm.list_custom_profiles() == []


class TestEndToEndPhase3Config:
    """Test de configuration Phase 3 complète."""

    def test_all_phase3_modules_initialize(self, tmp_path):
        """Tous les modules Phase 3 s'initialisent correctement."""
        # Each module should initialize without errors
        pi = PersistentInstructions(project_dir=tmp_path)
        assert not pi.is_configured()

        glossary = GlossaryEngine(project_dir=tmp_path, enabled=True)
        assert glossary.get_all_terms() == []

        personas = PersonaEngine(project_dir=tmp_path, enabled=True)
        assert personas.list_personas() == []

        lib_path = tmp_path / "lib.json"
        lib_path.write_text("[]", encoding="utf-8")
        templates = TemplateLibrary(library_path=lib_path)
        assert templates.list() == []

        feedback = FeedbackEngine(enabled=True)
        assert feedback.get_statistics()["total"] == 0

        journal_path = tmp_path / "hitl_journal.json"
        journal = HITLJournal(journal_path=journal_path)
        assert journal.get_statistics()["total"] == 0

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        overrides = MetadataOverrides(project_dir=project_dir)
        assert overrides.list_overrides() == []

        citations = CitationEngine(enabled=True)
        assert citations.get_cited_doc_ids() == set()
