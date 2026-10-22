"""Unit tests for Service Abstractions and Fallbacks."""

import pytest
from unittest.mock import MagicMock, patch


class TestInfrastructureDecoupling:
    """Infrastructure Decoupling and Research Dependency Isolation."""

    def test_feature_flag_isolation_clip_and_phase5(self, monkeypatch):
        from backend.model_loader import ModelStore

        monkeypatch.setenv("CLIP_ENABLED", "false")
        monkeypatch.setenv("PHASE5_ENABLED", "false")

        isolated_store = ModelStore()
        isolated_store.load()

        assert isolated_store.clip_loaded is False
        assert isolated_store.phase5_loaded is False
        assert isolated_store.combined_graph_model is not None
        assert isolated_store.is_loaded is True

    def test_clip_multimodal_scorer_instantiates_without_torch(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from trustshield_project.multimodal_scoring import CLIPMultimodalScorer
        scorer = CLIPMultimodalScorer(device=None)
        assert scorer.device == "cpu"
