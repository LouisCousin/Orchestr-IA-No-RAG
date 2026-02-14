"""Tests unitaires pour le module profile_manager."""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.core.profile_manager import ProfileManager, DEFAULT_PROFILES


@pytest.fixture
def profile_mgr(tmp_path):
    """Crée un ProfileManager avec un répertoire temporaire."""
    with patch("src.core.profile_manager.PROFILES_DIR", tmp_path / "profiles" / "default"):
        with patch("src.core.profile_manager.ROOT_DIR", tmp_path):
            mgr = ProfileManager()
            mgr.profiles_dir = tmp_path / "profiles" / "default"
            mgr._ensure_default_profiles()
            return mgr


class TestDefaultProfiles:
    def test_five_profiles_exist(self, profile_mgr):
        profiles = profile_mgr.list_profiles()
        assert len(profiles) == 5

    def test_profile_names(self, profile_mgr):
        profiles = profile_mgr.list_profiles()
        names = {p["name"] for p in profiles}
        expected = {
            "Rapport d'analyse",
            "Proposition de services",
            "Synthèse de veille",
            "Document de formation",
            "Compte rendu",
        }
        assert names == expected

    def test_each_profile_has_required_fields(self, profile_mgr):
        for profile_info in profile_mgr.list_profiles():
            profile = profile_mgr.load_profile(profile_info["id"])
            assert profile is not None
            assert "name" in profile
            assert "description" in profile
            assert "target_pages" in profile
            assert "generation" in profile
            assert "checkpoints" in profile
            assert "styling" in profile
            assert "persistent_instructions" in profile


class TestLoadProfile:
    def test_load_existing(self, profile_mgr):
        profile = profile_mgr.load_profile("rapport_analyse")
        assert profile is not None
        assert profile["name"] == "Rapport d'analyse"

    def test_load_nonexistent(self, profile_mgr):
        profile = profile_mgr.load_profile("nonexistent")
        assert profile is None


class TestGetProfileConfig:
    def test_config_keys(self, profile_mgr):
        config = profile_mgr.get_profile_config("rapport_analyse")
        assert "temperature" in config
        assert "max_tokens" in config
        assert "target_pages" in config
        assert "checkpoints" in config
        assert "styling" in config
        assert "persistent_instructions" in config

    def test_config_values(self, profile_mgr):
        config = profile_mgr.get_profile_config("compte_rendu")
        assert config["target_pages"] == 5
        assert config["temperature"] == 0.4

    def test_nonexistent_profile_returns_empty(self, profile_mgr):
        config = profile_mgr.get_profile_config("nonexistent")
        assert config == {}


class TestDefaultProfilesContent:
    def test_rapport_analyse(self):
        profile = DEFAULT_PROFILES["rapport_analyse"]
        assert profile["target_pages"] == 20
        assert "analytique" in profile["tone"]

    def test_proposition_services(self):
        profile = DEFAULT_PROFILES["proposition_services"]
        assert profile["target_pages"] == 15
        assert "persuasif" in profile["tone"]

    def test_synthese_veille(self):
        profile = DEFAULT_PROFILES["synthese_veille"]
        assert profile["target_pages"] == 10

    def test_document_formation(self):
        profile = DEFAULT_PROFILES["document_formation"]
        assert profile["target_pages"] == 25
        assert "pédagogique" in profile["tone"]

    def test_compte_rendu(self):
        profile = DEFAULT_PROFILES["compte_rendu"]
        assert profile["target_pages"] == 5
        assert "factuel" in profile["tone"]
