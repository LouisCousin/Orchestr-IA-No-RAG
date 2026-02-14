"""Tests unitaires pour le module checkpoint_manager."""

import pytest

from src.core.checkpoint_manager import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointResult,
    CheckpointType,
)


@pytest.fixture
def default_config():
    return CheckpointConfig()


@pytest.fixture
def full_config():
    return CheckpointConfig(
        after_plan_validation=True,
        after_corpus_acquisition=True,
        after_extraction=True,
        after_prompt_generation=True,
        after_generation=True,
        final_review=True,
    )


@pytest.fixture
def manager(default_config):
    return CheckpointManager(config=default_config)


class TestCheckpointConfig:
    def test_defaults(self, default_config):
        assert default_config.after_plan_validation is True
        assert default_config.final_review is True
        assert default_config.after_corpus_acquisition is False
        assert default_config.after_generation is False

    def test_is_enabled(self, default_config):
        assert default_config.is_enabled("after_plan_validation") is True
        assert default_config.is_enabled("after_generation") is False
        assert default_config.is_enabled("final_review") is True

    def test_serialization(self, full_config):
        data = full_config.to_dict()
        restored = CheckpointConfig.from_dict(data)
        assert restored.after_plan_validation == full_config.after_plan_validation
        assert restored.after_generation == full_config.after_generation
        assert restored.final_review == full_config.final_review


class TestCheckpointManager:
    def test_should_pause_enabled(self, manager):
        assert manager.should_pause("after_plan_validation") is True
        assert manager.should_pause("final_review") is True

    def test_should_pause_disabled(self, manager):
        assert manager.should_pause("after_generation") is False
        assert manager.should_pause("after_extraction") is False

    def test_create_checkpoint_when_enabled(self, manager):
        cp = manager.create_checkpoint(
            "after_plan_validation",
            content="Plan normalisé",
            section_id=None,
        )
        assert cp is not None
        assert cp["type"] == "after_plan_validation"
        assert manager.has_pending is True

    def test_create_checkpoint_when_disabled(self, manager):
        cp = manager.create_checkpoint(
            "after_generation",
            content="Section content",
        )
        assert cp is None
        assert manager.has_pending is False

    def test_resolve_checkpoint(self, manager):
        manager.create_checkpoint("after_plan_validation", content="Plan")
        result = manager.resolve_checkpoint("approved", user_comment="LGTM")
        assert result.action == "approved"
        assert result.user_comment == "LGTM"
        assert manager.has_pending is False

    def test_resolve_without_pending_raises(self, manager):
        with pytest.raises(RuntimeError):
            manager.resolve_checkpoint("approved")

    def test_history_tracking(self, manager):
        manager.create_checkpoint("after_plan_validation", content="Plan")
        manager.resolve_checkpoint("approved")

        manager.create_checkpoint("final_review", content="Review")
        manager.resolve_checkpoint("modified", modified_content="Updated")

        history = manager.get_history()
        assert len(history) == 2
        assert history[0]["action"] == "approved"
        assert history[1]["action"] == "modified"


class TestCheckpointResult:
    def test_auto_timestamp(self):
        result = CheckpointResult(
            checkpoint_type="test",
            action="approved",
        )
        assert result.timestamp != ""

    def test_with_section(self):
        result = CheckpointResult(
            checkpoint_type="after_generation",
            action="modified",
            section_id="1.1",
            original_content="original",
            modified_content="modified",
        )
        assert result.section_id == "1.1"
        assert result.modified_content == "modified"
