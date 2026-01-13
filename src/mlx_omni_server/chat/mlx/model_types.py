"""MLX Model types and management."""

import os
from pathlib import Path
from typing import Optional, Union

import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load, load_config

# Handle mlx_lm version compatibility: newer versions use hf_repo_to_path (returns Path),
# older versions (< 0.29) use get_model_path (returns Tuple[Path, Optional[str]])
try:
    from mlx_lm.utils import hf_repo_to_path as _get_model_path

    def get_model_path(model_id: str) -> Path:
        """Get model path (wrapper for newer mlx_lm versions)."""
        return _get_model_path(model_id)

except ImportError:
    from mlx_lm.utils import get_model_path as _get_model_path

    def get_model_path(model_id: str) -> Path:
        """Get model path (wrapper for older mlx_lm versions)."""
        result = _get_model_path(model_id)
        # Old version returns Tuple[Path, Optional[str]], extract just the Path
        if isinstance(result, tuple):
            return result[0]
        elif isinstance(result, Path):
            return result
        else:
            raise TypeError(f"Unexpected return type from get_model_path: {type(result)}")

from ...utils.logger import logger
from .tools.chat_template import ChatTemplate


def resolve_model_path(model_id: str) -> Path:
    """Resolve model path with custom path precedence.

    Checks MLX_OMNI_MODEL_PATH first, falls back to HuggingFace cache.
    Expected structure: {custom_path}/org/model-name

    Args:
        model_id: Model ID (e.g., "org/model-name" or local path)

    Returns:
        Path to the model directory

    Raises:
        ValueError: If model cannot be found in custom path and is invalid for HF cache
    """
    custom_model_path = os.environ.get("MLX_OMNI_MODEL_PATH")

    if custom_model_path:
        custom_path = Path(custom_model_path).expanduser().resolve()

        # Build the expected path for the model
        # Handles both "org/model" and "model" formats
        model_path = custom_path / model_id

        if model_path.exists() and model_path.is_dir():
            # Check if config.json exists to validate it's a proper model
            if (model_path / "config.json").exists():
                logger.debug(f"Resolved model {model_id} from custom path: {model_path}")
                return model_path
            else:
                logger.debug(
                    f"Model {model_id} found in custom path but missing config.json, falling back to HF cache"
                )
        else:
            logger.debug(
                f"Model {model_id} not found in custom path {custom_path}, falling back to HF cache"
            )

    # Fall back to HuggingFace cache via mlx_lm
    logger.debug(f"Resolving model {model_id} via HuggingFace cache")
    return get_model_path(model_id)


def load_mlx_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model_id: Optional[str] = None,
) -> "MLXModel":
    """Factory function to load MLX models.

    Args:
        model_id: Model name/path (HuggingFace model ID or local path)
        adapter_path: Optional path to LoRA adapter
        draft_model_id: Optional draft model name/path for speculative decoding

    Returns:
        MLXModel instance with loaded models

    Raises:
        ValueError: If model_id is invalid
        RuntimeError: If model loading fails
    """
    if not model_id or not model_id.strip():
        raise ValueError("model_id cannot be empty")

    model_id = model_id.strip()

    try:
        # Resolve model path first (checks custom path, then HF cache)
        model_path = resolve_model_path(model_id)

        # Load the main model using resolved path
        # Pass the filesystem path to load() to avoid unnecessary HF cache checks
        model, tokenizer = load(
            str(model_path),
            tokenizer_config={"trust_remote_code": True},
            adapter_path=adapter_path,
        )
        logger.info(f"Loaded model: {model_id} from {model_path}")

        # Load configuration and create chat tokenizer
        config = load_config(model_path)
        chat_template = ChatTemplate(config["model_type"], tokenizer)

        # Load draft model if specified
        draft_model = None
        draft_tokenizer = None
        if draft_model_id:
            try:
                # Resolve draft model path (checks custom path, then HF cache)
                draft_model_path = resolve_model_path(draft_model_id)

                # Load draft model using resolved path
                draft_model, draft_tokenizer = load(
                    str(draft_model_path),
                    tokenizer_config={"trust_remote_code": True},
                )

                # Check if vocabulary sizes match
                if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                    logger.warn(
                        f"Draft model({draft_model_id}) tokenizer does not match model tokenizer."
                    )

                logger.info(f"Loaded draft model: {draft_model_id} from {draft_model_path}")
            except Exception as e:
                logger.error(f"Failed to load draft model {draft_model_id}: {e}")
                # Continue without draft model
                draft_model = None
                draft_tokenizer = None

        return MLXModel(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            model=model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
        )

    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise RuntimeError(f"Model loading failed for {model_id}: {e}") from e


class MLXModel:
    """Simplified MLX model container.

    This class is a simple data container for loaded MLX models.
    For model management operations, create new instances rather than modifying existing ones.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str],
        draft_model_id: Optional[str],
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        chat_template: ChatTemplate,
        draft_model: Optional[nn.Module] = None,
        draft_tokenizer: Optional[TokenizerWrapper] = None,
    ):
        """Initialize MLX model container.

        This constructor is typically called by load_mlx_model() factory function.

        Args:
            model_id: Model name/path
            adapter_path: Path to LoRA adapter (if any)
            draft_model_id: Draft model name/path (if any)
            model: Loaded main model
            tokenizer: Loaded tokenizer
            chat_template: Chat template instance
            draft_model: Loaded draft model (optional)
            draft_tokenizer: Draft model tokenizer (optional)
        """
        # Model identification
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.draft_model_id = draft_model_id

        # Loaded model components
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

    @classmethod
    def load(
        cls,
        model_id: str,
        adapter_path: Optional[str] = None,
        draft_model_id: Optional[str] = None,
    ) -> "MLXModel":
        return load_mlx_model(model_id, adapter_path, draft_model_id)

    def __str__(self) -> str:
        """Return a string representation of the model for debugging."""
        parts = [f"model_id={self.model_id}"]
        if self.adapter_path:
            parts.append(f"adapter_path={self.adapter_path}")
        if self.draft_model_id:
            parts.append(f"draft_model_id={self.draft_model_id}")
        return f"MLXModel({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        """Check equality based on model configuration."""
        if not isinstance(other, MLXModel):
            return False
        return (
            self.model_id == other.model_id
            and self.adapter_path == other.adapter_path
            and self.draft_model_id == other.draft_model_id
        )

    def __hash__(self) -> int:
        """Hash based on model configuration for use as dict keys."""
        return hash((self.model_id, self.adapter_path, self.draft_model_id))

    def has_adapter(self) -> bool:
        """Check if this model has an adapter configured."""
        return self.adapter_path is not None

    def has_draft_model(self) -> bool:
        """Check if draft model is available."""
        return self.draft_model is not None and self.draft_tokenizer is not None

    def cleanup(self) -> None:
        """Explicitly release model resources.

        This method should be called before the MLXModel is destroyed to ensure
        that large model objects and any associated GPU/VRAM are properly released.
        """
        try:
            # Clear model references
            self.model = None
            self.tokenizer = None
            self.chat_template = None
            self.draft_model = None
            self.draft_tokenizer = None
            logger.debug(f"Cleaned up resources for model: {self.model_id}")
        except Exception as e:
            logger.error(f"Error during MLXModel cleanup: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup is called."""
        try:
            self.cleanup()
        except Exception:
            # Silently ignore exceptions during destruction
            pass
