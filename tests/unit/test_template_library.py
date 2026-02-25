"""Tests unitaires pour template_library.py."""

import json
import pytest
from pathlib import Path
from src.core.template_library import TemplateLibrary, TemplateVariableError, DEFAULT_TEMPLATES


@pytest.fixture
def lib(tmp_path):
    """Crée une bibliothèque vide (sans defaults)."""
    path = tmp_path / "library.json"
    # Créer un fichier vide pour éviter l'init des defaults
    path.write_text("[]", encoding="utf-8")
    return TemplateLibrary(library_path=path)


@pytest.fixture
def lib_with_defaults(tmp_path):
    """Crée une bibliothèque qui initialisera les templates par défaut."""
    path = tmp_path / "library.json"
    return TemplateLibrary(library_path=path)


class TestTemplateLibrary:
    def test_init_default_templates(self, lib_with_defaults):
        """Les templates par défaut sont créés au premier lancement."""
        templates = lib_with_defaults.list()
        assert len(templates) == len(DEFAULT_TEMPLATES)

    def test_create_template(self, lib):
        tid = lib.create(name="Mon template", content="Contenu {var1}", tags=["test"])
        assert tid is not None
        t = lib.get(tid)
        assert t["name"] == "Mon template"
        assert t["content"] == "Contenu {var1}"
        assert t["tags"] == ["test"]
        assert t["usage_count"] == 0

    def test_create_duplicate_name_raises(self, lib):
        lib.create(name="Unique", content="a")
        with pytest.raises(ValueError, match="existe déjà"):
            lib.create(name="unique", content="b")  # case-insensitive

    def test_get_nonexistent(self, lib):
        assert lib.get("nonexistent") is None

    def test_update_template(self, lib):
        tid = lib.create(name="Original", content="a")
        result = lib.update(tid, content="b", tags=["updated"])
        assert result["content"] == "b"
        assert result["tags"] == ["updated"]
        assert result["name"] == "Original"

    def test_update_protected_fields(self, lib):
        tid = lib.create(name="Protected", content="a")
        original = lib.get(tid)
        lib.update(tid, id="hacked", created_at="hacked")
        updated = lib.get(tid)
        assert updated["id"] == original["id"]
        assert updated["created_at"] == original["created_at"]

    def test_update_nonexistent(self, lib):
        assert lib.update("nonexistent", content="x") is None

    def test_delete_template(self, lib):
        tid = lib.create(name="ToDelete", content="a")
        assert lib.delete(tid) is True
        assert lib.get(tid) is None

    def test_delete_nonexistent(self, lib):
        assert lib.delete("nonexistent") is False

    def test_list_filter_by_tags(self, lib):
        lib.create(name="T1", content="a", tags=["alpha", "beta"])
        lib.create(name="T2", content="b", tags=["alpha"])
        lib.create(name="T3", content="c", tags=["gamma"])

        results = lib.list(tags=["alpha"])
        assert len(results) == 2
        results = lib.list(tags=["alpha", "beta"])
        assert len(results) == 1
        results = lib.list(tags=["delta"])
        assert len(results) == 0

    def test_list_search(self, lib):
        lib.create(name="Machine Learning", content="a", description="Deep learning guide")
        lib.create(name="Statistics", content="b", description="Basic stats")

        results = lib.list(search="machine")
        assert len(results) == 1
        assert results[0]["name"] == "Machine Learning"

        results = lib.list(search="stats")
        assert len(results) == 1

    def test_duplicate(self, lib):
        tid = lib.create(name="Original", content="Hello {name}", tags=["t1"])
        new_tid = lib.duplicate(tid, "Copy of Original")
        assert new_tid is not None
        copy = lib.get(new_tid)
        assert copy["name"] == "Copy of Original"
        assert copy["content"] == "Hello {name}"
        assert copy["tags"] == ["t1"]

    def test_duplicate_nonexistent(self, lib):
        assert lib.duplicate("nonexistent", "Copy") is None

    def test_resolve_all_provided(self, lib):
        tid = lib.create(
            name="Resolve Test",
            content="Bonjour {nom}, bienvenue à {lieu}.",
            variables=[
                {"name": "nom", "description": "Nom de la personne"},
                {"name": "lieu", "description": "Lieu"},
            ],
        )
        result = lib.resolve(tid, {"nom": "Alice", "lieu": "Paris"})
        assert result == "Bonjour Alice, bienvenue à Paris."

    def test_resolve_with_defaults(self, lib):
        tid = lib.create(
            name="Default Test",
            content="Sujet: {sujet}. Niveau: {niveau}.",
            variables=[
                {"name": "sujet", "description": "Sujet"},
                {"name": "niveau", "description": "Niveau", "default": "intermédiaire"},
            ],
        )
        result = lib.resolve(tid, {"sujet": "IA"})
        assert result == "Sujet: IA. Niveau: intermédiaire."

    def test_resolve_missing_required_raises(self, lib):
        tid = lib.create(
            name="Required Test",
            content="Hello {name}",
            variables=[{"name": "name", "description": "Name"}],
        )
        with pytest.raises(TemplateVariableError, match="name"):
            lib.resolve(tid, {})

    def test_resolve_nonexistent_template_raises(self, lib):
        with pytest.raises(ValueError, match="non trouvé"):
            lib.resolve("nonexistent", {})

    def test_resolve_increments_usage(self, lib):
        tid = lib.create(
            name="Usage Test",
            content="No vars",
            variables=[],
        )
        lib.resolve(tid)
        lib.resolve(tid)
        t = lib.get(tid)
        assert t["usage_count"] == 2

    def test_export_import(self, lib):
        tid = lib.create(name="Exportable", content="data", tags=["export"])
        exported = lib.export_template(tid)
        assert exported is not None

        new_tid = lib.import_template(exported)
        imported = lib.get(new_tid)
        # Name gets a suffix due to conflict
        assert imported["name"].startswith("Exportable")
        assert imported["content"] == "data"

    def test_import_name_conflict_resolution(self, lib):
        lib.create(name="Conflict", content="a")
        new_tid = lib.import_template({"name": "Conflict", "content": "b"})
        imported = lib.get(new_tid)
        assert imported["name"] == "Conflict (1)"

    def test_persistence(self, tmp_path):
        path = tmp_path / "lib.json"
        path.write_text("[]", encoding="utf-8")
        lib1 = TemplateLibrary(library_path=path)
        tid = lib1.create(name="Persisted", content="data")

        lib2 = TemplateLibrary(library_path=path)
        t = lib2.get(tid)
        assert t is not None
        assert t["name"] == "Persisted"
