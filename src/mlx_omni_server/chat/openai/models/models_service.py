import importlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

from huggingface_hub import CachedRepoInfo, scan_cache_dir

from ....utils.logger import logger
from .schema import Model, ModelDeletion, ModelList

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
}


class CustomModelInfo:
    """Information about a model in custom model path."""

    def __init__(self, model_id: str, model_path: Path, last_modified: float):
        self.repo_id = model_id
        self.model_path = model_path
        self.last_modified = last_modified
        self.repo_type = "model"


class ModelCacheScanner:
    """Scanner for finding and managing mlx-lm compatible models in the local cache."""

    def __init__(self):
        self._cache_info = None

    @property
    def cache_info(self):
        """Lazy load and cache the scan_cache_dir result"""
        if self._cache_info is None:
            self._cache_info = scan_cache_dir()
        return self._cache_info

    def _refresh_cache_info(self):
        """Force refresh the cache info"""
        self._cache_info = scan_cache_dir()

    def _get_model_classes(self, config: dict) -> Optional[Tuple[Type, Type]]:
        """
        Try to retrieve the model and model args classes based on the configuration.
        https://github.com/ml-explore/mlx-examples/blob/1e0766018494c46bc6078769278b8e2a360503dc/llms/mlx_lm/utils.py#L81

        Args:
            config (dict): The model configuration

        Returns:
            Optional tuple of (Model class, ModelArgs class) if model type is supported
        """
        try:
            model_type = config.get("model_type")
            model_type = MODEL_REMAPPING.get(model_type, model_type)
            if not model_type:
                return None

            # Try to import the model architecture module
            arch = importlib.import_module(f"mlx_lm.models.{model_type}")
            return arch.Model, arch.ModelArgs

        except ImportError:
            logger.debug(f"Model type {model_type} not supported by mlx-lm")
            return None
        except Exception as e:
            logger.warning(f"Error checking model compatibility: {str(e)}")
            return None

    def is_model_supported(self, config_data: Dict) -> bool:
        return self._get_model_classes(config_data) is not None

    def _scan_custom_path(
        self, custom_path: str
    ) -> List[Tuple[Union[CustomModelInfo, CachedRepoInfo], Dict]]:
        """Scan custom model path directory structure.

        Expected structure:
        {custom_path}/
          org1/
            model1/
              config.json
            model2/
              config.json
          org2/
            ...

        Args:
            custom_path: Path to custom models directory

        Returns:
            List of tuples containing (CustomModelInfo/CachedRepoInfo, config_dict)
        """
        custom_path_obj = Path(custom_path).expanduser().resolve()

        if not custom_path_obj.exists():
            logger.warning(f"Custom model path does not exist: {custom_path}")
            return []

        if not custom_path_obj.is_dir():
            logger.warning(f"Custom model path is not a directory: {custom_path}")
            return []

        supported_models = []

        try:
            # Walk through org directories
            for org_dir in custom_path_obj.iterdir():
                if not org_dir.is_dir():
                    continue

                # Walk through model directories within org
                for model_dir in org_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    config_file = model_dir / "config.json"
                    if not config_file.exists():
                        logger.debug(
                            f"Model directory missing config.json: {model_dir}"
                        )
                        continue

                    try:
                        with open(config_file, "r") as f:
                            config_data = json.load(f)

                        if not self.is_model_supported(config_data):
                            logger.debug(
                                f"Model {org_dir.name}/{model_dir.name} not supported by mlx-lm"
                            )
                            continue

                        model_id = f"{org_dir.name}/{model_dir.name}"
                        last_modified = model_dir.stat().st_mtime

                        model_info = CustomModelInfo(model_id, model_dir, last_modified)
                        supported_models.append((model_info, config_data))
                        logger.debug(f"Found custom model: {model_id}")

                    except Exception as e:
                        logger.error(
                            f"Error reading config.json for {org_dir.name}/{model_dir.name}: {str(e)}"
                        )

        except Exception as e:
            logger.error(f"Error scanning custom model path {custom_path}: {str(e)}")

        return supported_models

    def find_models_in_cache(self) -> List[Tuple[Union[CustomModelInfo, CachedRepoInfo], Dict]]:
        """
        Scan local cache for available models that are compatible with mlx-lm.

        Scans both custom model path (if configured) and HuggingFace cache.
        Custom path models take precedence if duplicates exist.

        Returns:
            List of tuples containing (CustomModelInfo/CachedRepoInfo, config_dict)
        """
        supported_models = []
        model_ids_seen = set()

        # Scan custom path first if configured
        custom_path = os.environ.get("MLX_OMNI_MODEL_PATH")
        if custom_path:
            custom_models = self._scan_custom_path(custom_path)
            for model_info, config_data in custom_models:
                supported_models.append((model_info, config_data))
                model_ids_seen.add(model_info.repo_id)

        # Scan HuggingFace cache
        for repo_info in self.cache_info.repos:
            if repo_info.repo_type != "model":
                continue

            # Skip if model already found in custom path
            if repo_info.repo_id in model_ids_seen:
                logger.debug(
                    f"Skipping HF cache model {repo_info.repo_id} (already found in custom path)"
                )
                continue

            first_revision = next(iter(repo_info.revisions), None)
            if not first_revision:
                continue

            config_file = next(
                (f for f in first_revision.files if f.file_name == "config.json"), None
            )
            if not config_file:
                continue

            try:
                with open(config_file.file_path, "r") as f:
                    config_data = json.load(f)
                if self.is_model_supported(config_data):
                    supported_models.append((repo_info, config_data))
            except Exception as e:
                logger.error(
                    f"Error reading config.json for {repo_info.repo_id}: {str(e)}"
                )

        return supported_models

    def get_model_info(
        self, model_id: str
    ) -> Optional[Tuple[Union[CustomModelInfo, CachedRepoInfo], Dict]]:
        # Check custom path first if configured
        custom_path = os.environ.get("MLX_OMNI_MODEL_PATH")
        if custom_path:
            custom_path_obj = Path(custom_path).expanduser().resolve()
            model_path = custom_path_obj / model_id

            if model_path.exists() and model_path.is_dir():
                config_file = model_path / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, "r") as f:
                            config_data = json.load(f)
                        if self.is_model_supported(config_data):
                            model_info = CustomModelInfo(
                                model_id, model_path, model_path.stat().st_mtime
                            )
                            return (model_info, config_data)
                        else:
                            logger.warning(
                                f"Model {model_id} found but not compatible with mlx-lm"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error reading config.json for {model_id}: {str(e)}"
                        )

        # Fall back to HuggingFace cache
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id and repo_info.repo_type == "model":
                first_revision = next(iter(repo_info.revisions), None)
                if not first_revision:
                    continue

                config_file = next(
                    (f for f in first_revision.files if f.file_name == "config.json"),
                    None,
                )
                if not config_file:
                    continue

                try:
                    with open(config_file.file_path, "r") as f:
                        config_data = json.load(f)
                    if self.is_model_supported(config_data):
                        return (repo_info, config_data)
                    else:
                        logger.warning(
                            f"Model {model_id} found but not compatible with mlx-lm"
                        )
                except Exception as e:
                    logger.error(
                        f"Error reading config.json for {repo_info.repo_id}: {str(e)}"
                    )

        return None

    def delete_model(self, model_id: str) -> bool:
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id:
                revision_hashes = [rev.commit_hash for rev in repo_info.revisions]
                if not revision_hashes:
                    return False

                try:
                    delete_strategy = self.cache_info.delete_revisions(*revision_hashes)
                    logger.info(
                        f"Model '{model_id}': Will free {delete_strategy.expected_freed_size_str}"
                    )
                    delete_strategy.execute()
                    logger.info(f"Model '{model_id}': Cache deletion completed")
                    self._refresh_cache_info()
                    return True
                except Exception as e:
                    logger.error(f"Error deleting model '{model_id}': {str(e)}")
                    raise

        return False


class ModelsService:
    def __init__(self):
        self.scanner = ModelCacheScanner()
        self.available_models = self._scan_models()

    def _scan_models(self) -> List[Tuple[CachedRepoInfo, Dict]]:
        """Scan local cache for available CausalLM models"""
        try:
            return self.scanner.find_models_in_cache()
        except Exception as e:
            print(f"Error scanning cache: {str(e)}")
            return []

    @staticmethod
    def _get_model_owner(model_id: str) -> str:
        """Extract owner from model ID (part before the /)"""
        return model_id.split("/")[0] if "/" in model_id else model_id

    def list_models(self, include_details: bool = False) -> ModelList:
        """List all available models"""
        models = []
        for repo_info, config_data in self.available_models:
            model_kwargs = {
                "id": repo_info.repo_id,
                "created": int(repo_info.last_modified),
                "owned_by": self._get_model_owner(repo_info.repo_id),
            }
            if include_details:
                model_kwargs["details"] = config_data
            model_instance = Model(**model_kwargs)
            models.append(model_instance)
        return ModelList(data=models)

    def get_model(
        self, model_id: str, include_details: bool = False
    ) -> Optional[Model]:
        """Get information about a specific model"""
        model_info = self.scanner.get_model_info(model_id)
        if model_info:
            repo_info, config_data = model_info
            model_kwargs = {
                "id": model_id,
                "created": int(repo_info.last_modified),
                "owned_by": self._get_model_owner(model_id),
            }
            if include_details:
                model_kwargs["details"] = config_data
            return Model(**model_kwargs)
        return None

    def delete_model(self, model_id: str) -> ModelDeletion:
        """Delete a model from local cache"""
        if not self.scanner.delete_model(model_id):
            raise ValueError(f"Model '{model_id}' not found in cache")

        self.available_models = self._scan_models()
        return ModelDeletion(id=model_id, deleted=True)
