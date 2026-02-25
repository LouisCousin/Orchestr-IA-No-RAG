"""Tests d'intégration pour le feedback loop complet."""

import pytest
from unittest.mock import MagicMock
from src.core.feedback_engine import FeedbackEngine, FeedbackEntry


@pytest.fixture
def provider():
    p = MagicMock()
    p.get_default_model.return_value = "gpt-4o"
    return p


@pytest.fixture
def engine(provider):
    return FeedbackEngine(
        provider=provider,
        enabled=True,
        min_diff_ratio=0.15,
    )


class TestFeedbackLoop:
    def test_full_feedback_cycle(self, engine, provider):
        """Cycle complet : détection → analyse → décision → ajustements."""
        analysis_response = '{"category": "style", "analysis": "Tone was too informal", "prompt_suggestion": "Use formal academic tone"}'
        provider.generate.return_value = MagicMock(content=analysis_response)

        original = "Bon ben voilà, l'IA c'est vraiment top pour faire des trucs."
        corrected = "L'intelligence artificielle constitue un outil puissant pour la réalisation de tâches complexes."

        # Step 1: Detect
        assert engine.detect_modification(original, corrected) is True

        # Step 2: Analyze
        entry = engine.analyze_modification("sec_1.1", original, corrected)
        assert entry is not None
        assert entry.category == "style"
        assert entry.suggestion == "Use formal academic tone"
        assert len(engine.pending_suggestions) == 1

        # Step 3: Accept
        engine.accept_suggestion(entry)
        assert len(engine.pending_suggestions) == 0
        assert len(engine.history) == 1

        # Step 4: Active adjustments
        adjustments = engine.get_active_adjustments()
        assert len(adjustments) == 1
        assert "formal academic tone" in adjustments[0]

    def test_multiple_feedback_entries(self, engine, provider):
        """Plusieurs feedbacks s'accumulent correctement."""
        responses = [
            '{"category": "style", "analysis": "Tone issue", "prompt_suggestion": "Be formal"}',
            '{"category": "contenu", "analysis": "Missing data", "prompt_suggestion": "Include statistics"}',
            '{"category": "structure", "analysis": "Poor organization", "prompt_suggestion": "Use headings"}',
        ]
        call_idx = 0

        def side_effect(*args, **kwargs):
            nonlocal call_idx
            resp = MagicMock(content=responses[call_idx])
            call_idx += 1
            return resp

        provider.generate.side_effect = side_effect

        pairs = [
            ("sec_1", "informal text here", "Formal text here instead of the informal one"),
            ("sec_2", "data not included in this text", "Data: 42% of users prefer the new approach"),
            ("sec_3", "unstructured messy text", "## Section Title\nOrganized text with clear structure"),
        ]

        entries = []
        for sec_id, original, corrected in pairs:
            entry = engine.analyze_modification(sec_id, original, corrected)
            if entry:
                entries.append(entry)

        assert len(entries) == 3
        assert len(engine.pending_suggestions) == 3

        # Accept first, reject second, modify third
        engine.accept_suggestion(entries[0])
        engine.reject_suggestion(entries[1])
        engine.modify_suggestion(entries[2], "Use H2 headings for main sections")

        assert len(engine.pending_suggestions) == 0
        assert len(engine.history) == 3

        # Only accepted and modified are active
        adjustments = engine.get_active_adjustments()
        assert len(adjustments) == 2
        assert "Be formal" in adjustments
        assert "Use H2 headings for main sections" in adjustments

    def test_statistics_tracking(self, engine, provider):
        """Les statistiques reflètent les décisions."""
        analysis_response = '{"category": "style", "analysis": "Test", "prompt_suggestion": "Suggestion"}'
        provider.generate.return_value = MagicMock(content=analysis_response)

        # Create and process 4 entries
        for i in range(4):
            original = f"Original text version {i} that is quite different from the corrected version"
            corrected = f"Completely rewritten corrected text number {i} with substantial changes made"
            entry = engine.analyze_modification(f"sec_{i}", original, corrected)
            if entry:
                if i < 2:
                    engine.accept_suggestion(entry)
                elif i == 2:
                    engine.reject_suggestion(entry)
                else:
                    engine.modify_suggestion(entry, "Modified")

        stats = engine.get_statistics()
        assert stats["total"] == 4
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["modified"] == 1
        assert stats["acceptance_rate"] == 0.75
        assert "style" in stats["categories"]

    def test_below_threshold_no_analysis(self, engine, provider):
        """Modifications mineures ne déclenchent pas l'analyse."""
        original = "Le texte est bien structuré."
        corrected = "Le texte est bien structuré!"  # Minor punctuation change

        assert engine.detect_modification(original, corrected) is False
        entry = engine.analyze_modification("sec_1", original, corrected)
        assert entry is None
        provider.generate.assert_not_called()

    def test_api_failure_graceful(self, engine, provider):
        """Erreur API ne casse pas le pipeline."""
        provider.generate.side_effect = Exception("API timeout")

        original = "Completely different original text that has been substantially changed"
        corrected = "The corrected version is totally new and rewritten from scratch"

        entry = engine.analyze_modification("sec_1", original, corrected)
        assert entry is None
        assert len(engine.pending_suggestions) == 0

    def test_feedback_entry_serialization(self):
        """FeedbackEntry se sérialise correctement."""
        entry = FeedbackEntry(
            section_id="sec_1",
            category="contenu",
            suggestion="Add sources",
            decision="accepted",
            original_prompt="Long prompt " * 100,
            analysis="Content was missing references",
        )
        d = entry.to_dict()
        assert d["section_id"] == "sec_1"
        assert d["category"] == "contenu"
        assert len(d["original_prompt"]) <= 500
        assert d["timestamp"] is not None
