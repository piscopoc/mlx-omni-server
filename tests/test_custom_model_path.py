"""Tests for custom model path feature."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_omni_server.chat.mlx.model_types import resolve_model_path
from mlx_omni_server.chat.openai.models.models_service import (
    CustomModelInfo,
    ModelCacheScanner,
)


@pytest.fixture
def temp_custom_path():
    """Create a temporary directory with mock model structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = Path(tmpdir)

        # Create model directory structure: org/model/config.json
        models = [
            ("mlx-community", "Llama-3-8B-Instruct"),
            ("mlx-community", "Qwen2-7B"),
            ("Qwen", "Qwen2.5-72B-Instruct"),
        ]

        for org, model in models:
            model_dir = custom_path / org / model
            model_dir.mkdir(parents=True, exist_ok=True)

            # Create a minimal config.json for a supported model type
            config = {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            }

            with open(model_dir / "config.json", "w") as f:
                json.dump(config, f)

            # Create dummy model files
            (model_dir / "model.safetensors").touch()
            (model_dir / "tokenizer.json").touch()

        yield custom_path


@pytest.fixture
def temp_custom_path_with_unsupported():
    """Create a temporary directory with both supported and unsupported models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = Path(tmpdir)

        # Create supported model
        supported_dir = custom_path / "mlx-community" / "Llama-3-8B"
        supported_dir.mkdir(parents=True, exist_ok=True)
        with open(supported_dir / "config.json", "w") as f:
            json.dump({"model_type": "llama", "hidden_size": 4096}, f)

        # Create unsupported model (no valid model_type)
        unsupported_dir = custom_path / "test-org" / "unsupported-model"
        unsupported_dir.mkdir(parents=True, exist_ok=True)
        with open(unsupported_dir / "config.json", "w") as f:
            json.dump({"model_type": "unknown_type_xyz"}, f)

        # Create directory without config.json
        invalid_dir = custom_path / "test-org" / "no-config"
        invalid_dir.mkdir(parents=True, exist_ok=True)

        yield custom_path


class TestResolveModelPath:
    """Tests for resolve_model_path function."""

    def test_resolve_from_custom_path(self, temp_custom_path):
        """Test resolving model from custom path."""
        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            # Test resolving existing model
            model_path = resolve_model_path("mlx-community/Llama-3-8B-Instruct")
            assert model_path.exists()
            assert (model_path / "config.json").exists()
            assert model_path.name == "Llama-3-8B-Instruct"

    def test_resolve_fallback_to_hf_cache(self, temp_custom_path):
        """Test fallback to HF cache when model not in custom path."""
        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            # Mock get_model_path to avoid actual HF cache access
            with patch(
                "mlx_omni_server.chat.mlx.model_types.get_model_path"
            ) as mock_get_path:
                mock_get_path.return_value = Path("/hf/cache/model")

                # Try to resolve a model not in custom path
                result = resolve_model_path("unknown/model")

                # Should fall back to get_model_path
                mock_get_path.assert_called_once_with("unknown/model")
                assert result == Path("/hf/cache/model")

    def test_resolve_without_custom_path(self):
        """Test resolving without custom path uses HF cache."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLX_OMNI_MODEL_PATH", None)

            with patch(
                "mlx_omni_server.chat.mlx.model_types.get_model_path"
            ) as mock_get_path:
                mock_get_path.return_value = Path("/hf/cache/model")

                result = resolve_model_path("some/model")

                mock_get_path.assert_called_once_with("some/model")
                assert result == Path("/hf/cache/model")

    def test_resolve_invalid_custom_path(self, temp_custom_path):
        """Test resolve with invalid custom path falls back gracefully."""
        invalid_path = "/nonexistent/path/that/does/not/exist"

        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": invalid_path}):
            with patch(
                "mlx_omni_server.chat.mlx.model_types.get_model_path"
            ) as mock_get_path:
                mock_get_path.return_value = Path("/hf/cache/model")

                result = resolve_model_path("any/model")

                # Should fall back to HF cache
                mock_get_path.assert_called_once_with("any/model")


class TestModelCacheScannerCustomPath:
    """Tests for ModelCacheScanner custom path functionality."""

    def test_scan_custom_path(self, temp_custom_path):
        """Test scanning custom model path."""
        scanner = ModelCacheScanner()

        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            # Mock cache_info to avoid HF cache
            with patch.object(scanner, "cache_info") as mock_cache:
                mock_cache.repos = []

                results = scanner.find_models_in_cache()

                # Should find 3 models from custom path
                assert len(results) == 3

                model_ids = [info.repo_id for info, _ in results]
                assert "mlx-community/Llama-3-8B-Instruct" in model_ids
                assert "mlx-community/Qwen2-7B" in model_ids
                assert "Qwen/Qwen2.5-72B-Instruct" in model_ids

    def test_scan_custom_path_filters_unsupported(
        self, temp_custom_path_with_unsupported
    ):
        """Test scanning custom path filters out unsupported models."""
        scanner = ModelCacheScanner()

        with patch.dict(
            os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path_with_unsupported)}
        ):
            with patch.object(scanner, "cache_info") as mock_cache:
                mock_cache.repos = []

                results = scanner.find_models_in_cache()

                # Should only find 1 supported model
                assert len(results) == 1
                model_id = results[0][0].repo_id
                assert model_id == "mlx-community/Llama-3-8B"

    def test_custom_path_precedence_over_hf_cache(
        self, temp_custom_path, mocker
    ):
        """Test custom path models take precedence over HF cache."""
        scanner = ModelCacheScanner()

        # Mock HF cache with a duplicate model
        mock_repo_info = mocker.Mock()
        mock_repo_info.repo_id = "mlx-community/Llama-3-8B-Instruct"
        mock_repo_info.repo_type = "model"
        mock_repo_info.last_modified = 100  # Older timestamp

        mock_revision = mocker.Mock()
        mock_config_file = mocker.Mock()
        mock_config_file.file_name = "config.json"
        mock_config_file.file_path = "/tmp/fake/config.json"
        mock_revision.files = [mock_config_file]
        mock_repo_info.revisions = [mock_revision]

        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            with patch.object(scanner, "cache_info") as mock_cache:
                mock_cache.repos = [mock_repo_info]

                # Mock the file reading for HF cache
                with patch(
                    "builtins.open",
                    mocker.mock_open(json.dumps({"model_type": "llama"})),
                ):
                    results = scanner.find_models_in_cache()

                    # Find the Llama-3-8B-Instruct model
                    llama_model = next(
                        (r for r in results if "Llama-3-8B-Instruct" in r[0].repo_id),
                        None,
                    )

                    assert llama_model is not None
                    # Should be CustomModelInfo (from custom path), not HF cache
                    assert isinstance(llama_model[0], CustomModelInfo)

    def test_get_model_info_from_custom_path(self, temp_custom_path):
        """Test getting model info from custom path."""
        scanner = ModelCacheScanner()

        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            with patch.object(scanner, "cache_info") as mock_cache:
                mock_cache.repos = []

                model_info = scanner.get_model_info("mlx-community/Llama-3-8B-Instruct")

                assert model_info is not None
                assert isinstance(model_info[0], CustomModelInfo)
                assert model_info[0].repo_id == "mlx-community/Llama-3-8B-Instruct"
                assert isinstance(model_info[1], dict)

    def test_get_model_info_missing_returns_none(self, temp_custom_path):
        """Test getting missing model info returns None."""
        scanner = ModelCacheScanner()

        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            with patch.object(scanner, "cache_info") as mock_cache:
                mock_cache.repos = []

                model_info = scanner.get_model_info("nonexistent/model")

                assert model_info is None

    def test_custom_model_info_attributes(self, temp_custom_path):
        """Test CustomModelInfo has correct attributes."""
        model_path = temp_custom_path / "mlx-community" / "Llama-3-8B-Instruct"
        model_info = CustomModelInfo(
            "mlx-community/Llama-3-8B-Instruct", model_path, 1234567890
        )

        assert model_info.repo_id == "mlx-community/Llama-3-8B-Instruct"
        assert model_info.repo_type == "model"
        assert model_info.last_modified == 1234567890
        assert model_info.model_path == model_path

    def test_no_custom_path_uses_hf_cache(self, mocker):
        """Test that without custom path, only HF cache is scanned."""
        scanner = ModelCacheScanner()

        mock_repo_info = mocker.Mock()
        mock_repo_info.repo_id = "gpt2"
        mock_repo_info.repo_type = "model"
        mock_repo_info.last_modified = 1234567890

        mock_revision = mocker.Mock()
        mock_config_file = mocker.Mock()
        mock_config_file.file_name = "config.json"
        mock_config_file.file_path = "/tmp/config.json"
        mock_revision.files = [mock_config_file]
        mock_repo_info.revisions = [mock_revision]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLX_OMNI_MODEL_PATH", None)

            with patch.object(scanner, "cache_info") as mock_cache:
                mock_cache.repos = [mock_repo_info]

                with patch(
                    "builtins.open",
                    mocker.mock_open(json.dumps({"model_type": "gpt2"})),
                ):
                    results = scanner.find_models_in_cache()

                    # Should find model from HF cache
                    assert len(results) == 1
                    assert results[0][0].repo_id == "gpt2"
                    # Should NOT be CustomModelInfo
                    assert not isinstance(results[0][0], CustomModelInfo)


class TestModelsService:
    """Tests for ModelsService with custom paths."""

    def test_list_models_includes_custom_path(self, temp_custom_path, mocker):
        """Test list_models includes models from custom path."""
        from mlx_omni_server.chat.openai.models.models_service import ModelsService

        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            # Mock the HF cache to be empty
            with patch(
                "mlx_omni_server.chat.openai.models.models_service.scan_cache_dir"
            ) as mock_scan:
                mock_scan.return_value.repos = []

                service = ModelsService()
                model_list = service.list_models()

                # Should have 3 models from custom path
                assert len(model_list.data) == 3

                model_ids = [m.id for m in model_list.data]
                assert "mlx-community/Llama-3-8B-Instruct" in model_ids
                assert "mlx-community/Qwen2-7B" in model_ids
                assert "Qwen/Qwen2.5-72B-Instruct" in model_ids

    def test_anthropic_service_includes_custom_path(self, temp_custom_path, mocker):
        """Test AnthropicModelsService includes models from custom path."""
        from mlx_omni_server.chat.anthropic.models_service import (
            AnthropicModelsService,
        )

        with patch.dict(os.environ, {"MLX_OMNI_MODEL_PATH": str(temp_custom_path)}):
            with patch(
                "mlx_omni_server.chat.openai.models.models_service.scan_cache_dir"
            ) as mock_scan:
                mock_scan.return_value.repos = []

                service = AnthropicModelsService()
                model_list = service.list_models()

                # Should have 3 models from custom path
                assert len(model_list.data) == 3

                model_ids = [m.id for m in model_list.data]
                assert "mlx-community/Llama-3-8B-Instruct" in model_ids
