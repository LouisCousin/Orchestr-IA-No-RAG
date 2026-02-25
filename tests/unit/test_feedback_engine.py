"""Tests unitaires pour feedback_engine.py."""

import pytest
from unittest.mock import MagicMock, patch
from src.core.feedback_engine import FeedbackEngine, FeedbackEntry


@pytest.fixture
def engine():
    return FeedbackEngine(provider=None, enabled=True, min_diff_ratio=0.15)


@pytest.fixture
def engine_with_provider():
    provider = MagicMock()
    provider.get_default_model.return_value = "gpt-4o"
    return FeedbackEngine(provider=provider, enabled=True, min_diff_ratio=0.15)


class TestFeedbackEntry:
    def test_to_dict(self):
        entry = FeedbackEntry(
            section_id="sec_1",
            category="style",
            suggestion="Use more formal tone",
            decision="accepted",
            original_prompt="old prompt " * 200,  # test truncation
            analysis="The tone was too casual",
        )
        d = entry.to_dict()
        assert d["section_id"] == "sec_1"
        assert d["category"] == "style"
        assert d["decision"] == "accepted"
        assert len(d["original_prompt"]) <= 500
        assert d["timestamp"] is not None

    def test_default_decision(self):
        entry = FeedbackEntry(section_id="s", category="c", suggestion="s")
        assert entry.decision == "pending"


class TestFeedbackEngine:
    def test_detect_modification_significant(self, engine):
        original = "Ceci est un texte assez long pour être comparé de manière significative."
        corrected = "Ceci est un texte complètement différent qui a été entièrement remanié."
        assert engine.detect_modification(original, corrected) is True

    def test_detect_modification_minor(self, engine):
        original = "Ceci est un texte qui ne change presque pas du tout."
        corrected = "Ceci est un texte qui ne change presque pas du tout!"
        assert engine.detect_modification(original, corrected) is False

    def test_detect_modification_empty(self, engine):
        assert engine.detect_modification("", "text") is False
        assert engine.detect_modification("text", "") is False
        assert engine.detect_modification("", "") is False

    def test_detect_modification_identical(self, engine):
        text = "Identical text"
        assert engine.detect_modification(text, text) is False

    def test_analyze_modification_disabled(self):
        engine = FeedbackEngine(provider=MagicMock(), enabled=False)
        result = engine.analyze_modification("sec_1", "old", "new")
        assert result is None

    def test_analyze_modification_no_provider(self, engine):
        result = engine.analyze_modification("sec_1", "old", "new")
        assert result is None

    def test_analyze_modification_below_threshold(self, engine_with_provider):
        original = "Same text"
        corrected = "Same text!"
        result = engine_with_provider.analyze_modification("sec_1", original, corrected)
        assert result is None

    def test_analyze_modification_success(self, engine_with_provider):
        response_json = '{"category": "style", "analysis": "Tone changed", "prompt_suggestion": "Use formal tone"}'
        engine_with_provider.provider.generate.return_value = MagicMock(content=response_json)

        original = "Ceci est un texte très informel et relâché."
        corrected = "Ce document présente une analyse formelle et structurée du sujet abordé."

        result = engine_with_provider.analyze_modification("sec_1", original, corrected)
        assert result is not None
        assert result.category == "style"
        assert result.suggestion == "Use formal tone"
        assert len(engine_with_provider.pending_suggestions) == 1

    def test_analyze_modification_invalid_json(self, engine_with_provider):
        engine_with_provider.provider.generate.return_value = MagicMock(content="not json")

        original = "Completely different original text here."
        corrected = "Totally new and rewritten corrected version."

        result = engine_with_provider.analyze_modification("sec_1", original, corrected)
        assert result is None

    def test_analyze_modification_exception(self, engine_with_provider):
        engine_with_provider.provider.generate.side_effect = Exception("API error")

        original = "Completely different original text here."
        corrected = "Totally new and rewritten corrected version."

        result = engine_with_provider.analyze_modification("sec_1", original, corrected)
        assert result is None

    def test_accept_suggestion(self, engine):
        entry = FeedbackEntry(section_id="s1", category="style", suggestion="Be formal")
        engine._pending_suggestions.append(entry)
        engine.accept_suggestion(entry)
        assert entry.decision == "accepted"
        assert len(engine.history) == 1
        assert len(engine.pending_suggestions) == 0

    def test_reject_suggestion(self, engine):
        entry = FeedbackEntry(section_id="s1", category="style", suggestion="Suggestion")
        engine._pending_suggestions.append(entry)
        engine.reject_suggestion(entry)
        assert entry.decision == "rejected"
        assert len(engine.history) == 1
        assert len(engine.pending_suggestions) == 0

    def test_modify_suggestion(self, engine):
        entry = FeedbackEntry(section_id="s1", category="style", suggestion="Original")
        engine._pending_suggestions.append(entry)
        engine.modify_suggestion(entry, "Modified suggestion text")
        assert entry.decision == "modified"
        assert entry.modified_prompt == "Modified suggestion text"
        assert len(engine.history) == 1

    def test_get_active_adjustments(self, engine):
        e1 = FeedbackEntry(section_id="s1", category="style", suggestion="Accepted suggestion")
        e1.decision = "accepted"
        engine._history.append(e1)

        e2 = FeedbackEntry(section_id="s2", category="contenu", suggestion="Rejected")
        e2.decision = "rejected"
        engine._history.append(e2)

        e3 = FeedbackEntry(section_id="s3", category="structure", suggestion="Original")
        e3.decision = "modified"
        e3.modified_prompt = "Modified version"
        engine._history.append(e3)

        adjustments = engine.get_active_adjustments()
        assert len(adjustments) == 2
        assert "Accepted suggestion" in adjustments
        assert "Modified version" in adjustments

    def test_get_statistics_empty(self, engine):
        stats = engine.get_statistics()
        assert stats["total"] == 0
        assert stats["acceptance_rate"] == 0.0

    def test_get_statistics_with_data(self, engine):
        for decision, cat in [("accepted", "style"), ("rejected", "contenu"), ("modified", "style"), ("accepted", "structure")]:
            e = FeedbackEntry(section_id="s", category=cat, suggestion="s")
            e.decision = decision
            engine._history.append(e)

        stats = engine.get_statistics()
        assert stats["total"] == 4
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["modified"] == 1
        assert stats["acceptance_rate"] == 0.75
        assert stats["categories"]["style"] == 2

    def test_levenshtein_ratio_identical(self):
        assert FeedbackEngine._levenshtein_ratio("abc", "abc") == pytest.approx(0.0, abs=0.01)

    def test_levenshtein_ratio_different(self):
        ratio = FeedbackEngine._levenshtein_ratio("hello", "world")
        assert ratio > 0.5

    def test_parse_json_valid(self):
        result = FeedbackEngine._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_invalid(self):
        assert FeedbackEngine._parse_json("not json") == {}

    def test_parse_json_with_markdown(self):
        result = FeedbackEngine._parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}
