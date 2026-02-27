"""Tests unitaires pour le DAG de dépendances (Phase 7)."""

import pytest

from src.core.multi_agent_orchestrator import build_dag, get_ready_sections


class TestBuildDag:
    def test_no_dependencies(self):
        """Toutes les sections sont indépendantes."""
        deps = {"s01": [], "s02": [], "s03": []}
        dag = build_dag(deps)
        assert dag == deps

    def test_linear_dependencies(self):
        """Dépendances linéaires : s01 → s02 → s03."""
        deps = {"s01": [], "s02": ["s01"], "s03": ["s02"]}
        dag = build_dag(deps)
        assert dag == deps

    def test_complex_dependencies(self):
        """DAG complexe avec branches parallèles."""
        deps = {
            "s01": [],
            "s02": [],
            "s03": ["s01"],
            "s04": ["s01", "s02"],
            "s05": ["s03", "s04"],
        }
        dag = build_dag(deps)
        assert dag == deps

    def test_cycle_detection(self):
        """Cycle direct entre deux sections."""
        deps = {"s01": ["s02"], "s02": ["s01"]}
        with pytest.raises(ValueError, match="dépendances cycliques"):
            build_dag(deps)

    def test_cycle_detection_indirect(self):
        """Cycle indirect : s01 → s02 → s03 → s01."""
        deps = {"s01": ["s03"], "s02": ["s01"], "s03": ["s02"]}
        with pytest.raises(ValueError, match="dépendances cycliques"):
            build_dag(deps)

    def test_empty_dag(self):
        """DAG vide."""
        dag = build_dag({})
        assert dag == {}

    def test_single_section(self):
        """Une seule section."""
        deps = {"s01": []}
        dag = build_dag(deps)
        assert dag == {"s01": []}

    def test_missing_dependency_reference(self):
        """Dépendance vers une section non déclarée — pas de cycle."""
        deps = {"s01": [], "s02": ["s99"]}  # s99 n'existe pas comme clé
        dag = build_dag(deps)  # Ne doit pas lever d'erreur
        assert dag == deps


class TestGetReadySections:
    def test_all_ready(self):
        """Toutes les sections sont prêtes (aucune dépendance)."""
        dag = {"s01": [], "s02": [], "s03": []}
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert set(ready) == {"s01", "s02", "s03"}

    def test_none_ready(self):
        """Aucune section n'est prête (dépendances non résolues)."""
        dag = {"s01": ["s02"], "s02": ["s01"]}
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert ready == []  # Cycle — rien n'est prêt

    def test_partial_ready(self):
        """Certaines sections débloquées après complétion."""
        dag = {
            "s01": [],
            "s02": [],
            "s03": ["s01"],
            "s04": ["s01", "s02"],
        }
        # Initialement, s01 et s02 sont prêtes
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert set(ready) == {"s01", "s02"}

        # Après complétion de s01
        ready = get_ready_sections(dag, completed={"s01"}, in_progress=set())
        assert "s02" in ready
        assert "s03" in ready  # s03 dépend seulement de s01

        # s04 n'est pas prête (attend s02)
        assert "s04" not in ready

    def test_excludes_in_progress(self):
        """Les sections en cours ne sont pas relancées."""
        dag = {"s01": [], "s02": []}
        ready = get_ready_sections(dag, completed=set(), in_progress={"s01"})
        assert ready == ["s02"]

    def test_excludes_completed(self):
        """Les sections complétées ne sont pas relancées."""
        dag = {"s01": [], "s02": []}
        ready = get_ready_sections(dag, completed={"s01"}, in_progress=set())
        assert ready == ["s02"]

    def test_all_completed(self):
        """Toutes les sections sont complétées."""
        dag = {"s01": [], "s02": []}
        ready = get_ready_sections(dag, completed={"s01", "s02"}, in_progress=set())
        assert ready == []

    def test_sequential_unlock(self):
        """Déblocage séquentiel : s01 → s02 → s03."""
        dag = {"s01": [], "s02": ["s01"], "s03": ["s02"]}

        # Étape 1 : seul s01 est prêt
        ready = get_ready_sections(dag, completed=set(), in_progress=set())
        assert ready == ["s01"]

        # Étape 2 : s01 terminé → s02 débloqué
        ready = get_ready_sections(dag, completed={"s01"}, in_progress=set())
        assert ready == ["s02"]

        # Étape 3 : s02 terminé → s03 débloqué
        ready = get_ready_sections(dag, completed={"s01", "s02"}, in_progress=set())
        assert ready == ["s03"]
