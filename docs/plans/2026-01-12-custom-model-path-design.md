# Custom Model Path Configuration

**Date:** 2026-01-12
**Status:** Approved

## Summary

Add a configurable default model path that serves as the primary location for loading models, with HuggingFace's cache as a fallback. This enables storing models on external drives or centralized directories while maintaining compatibility with existing workflows.

## Motivation

- **Storage management:** HuggingFace's default cache can fill up the main drive; users want models on external SSDs or different partitions
- **Centralized organization:** Keep all models in a specific directory for easier backup, organization, or sharing across projects

## Configuration

Two ways to set the custom model path:

| Method | Example |
|--------|---------|
| Environment variable | `MLX_OMNI_MODEL_PATH=/Volumes/LLMS/models` |
| CLI argument | `--model-path /Volumes/LLMS/models` |

**Precedence (highest to lowest):**

1. CLI argument `--model-path`
2. Environment variable `MLX_OMNI_MODEL_PATH`
3. Default: None (use HuggingFace cache only, current behavior)

### Usage Examples

```bash
# Via environment variable
export MLX_OMNI_MODEL_PATH="/Volumes/LLMS/models"
mlx-omni-server --port 8080

# Via CLI (overrides env var if both set)
mlx-omni-server --model-path /Volumes/LLMS/models --port 8080
```

## Directory Structure

Models are organized in an `org/model-name` hierarchy matching HuggingFace repository naming:

```
/Volumes/LLMS/models/
├── mlx-community/
│   ├── Llama-3-8B-Instruct/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer.json
│   └── Qwen2-7B/
│       └── ...
└── Qwen/
    └── Qwen2.5-72B-Instruct/
        └── ...
```

## Model Resolution Logic

### Lookup Flow

When a model is requested (e.g., `mlx-community/Llama-3-8B-Instruct`):

```
1. If custom model path is configured:
   → Look for: {custom_path}/{org}/{model}/config.json
   → If found and valid: load from custom path
   → If not found: continue to step 2

2. Fall back to HuggingFace cache:
   → Use existing mlx_lm.load() behavior
   → Downloads if not cached (existing behavior)
```

### Validation

A model directory is considered valid if:
- The directory exists
- Contains a `config.json` file
- The config indicates an mlx-lm compatible model type (same validation as current `ModelCacheScanner`)

### Path Parsing

The model ID `org/model-name` maps directly to the filesystem:
- `mlx-community/Llama-3-8B-Instruct` → `{custom_path}/mlx-community/Llama-3-8B-Instruct/`

Edge cases:
- **No org prefix** (e.g., `gpt2`): Look for `{custom_path}/gpt2/` (flat, no org subdirectory)
- **Nested model names** (e.g., `org/sub/model`): Treated as `{custom_path}/org/sub/model/` (preserves full path structure)

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Custom path doesn't exist | Log warning at startup, proceed with HF-only mode |
| Model not in custom path | Silent fallback to HF cache (normal operation) |
| Model not in either location | Existing behavior (HF downloads or errors if offline) |

## Model Listing (`/v1/models` Endpoint)

### Merged Listing Behavior

When the models endpoint is called, scan both locations and merge results:

```
1. Scan custom model path (if configured):
   → Walk {custom_path}/*/* for org/model structure
   → Walk {custom_path}/* for flat models (no org)
   → Validate each: must have config.json with compatible model type
   → Collect as: {org}/{model} or {model}

2. Scan HuggingFace cache:
   → Use existing ModelCacheScanner.find_models_in_cache()
   → Returns list of cached model IDs

3. Merge and deduplicate:
   → Combine both lists
   → If same model ID exists in both, keep custom path version
   → Sort alphabetically by model ID
```

### Response Format

No changes to the existing OpenAI-compatible response format:

```json
{
  "object": "list",
  "data": [
    {"id": "mlx-community/Llama-3-8B-Instruct", "object": "model", ...},
    {"id": "Qwen/Qwen2.5-7B", "object": "model", ...}
  ]
}
```

The source (custom vs HF cache) is not exposed in the API response - it's an implementation detail.

## Implementation

### Files to Modify

1. **`src/mlx_omni_server/main.py`**
   - Add `--model-path` argument to `build_parser()`
   - Read `MLX_OMNI_MODEL_PATH` env var
   - Store resolved path in app state or module-level config

2. **`src/mlx_omni_server/chat/mlx/model_types.py`**
   - Modify `load_mlx_model()` to check custom path first
   - Add helper function `resolve_model_path(model_id: str, custom_path: Optional[str]) -> Optional[Path]`

3. **`src/mlx_omni_server/chat/openai/models/models_service.py`**
   - Add new class or method to scan custom model path
   - Modify `ModelsService` to merge results from both scanners
   - Deduplicate with custom path taking precedence

4. **`src/mlx_omni_server/chat/anthropic/models_service.py`**
   - Same changes as OpenAI models service (likely shares logic)

### New Code

One new utility function (in `model_types.py` or a new `config.py`):

```python
def get_model_path_config() -> Optional[Path]:
    """Returns the configured custom model path, or None."""
    # Access from app state or module config
```

### No Changes Needed

- `wrapper_cache.py` - Cache keys remain the same (model_id based)
- `chat_generator.py` - Unchanged, receives resolved paths
- Embeddings/Images/TTS/STT - Out of scope for this iteration

## Testing

### Manual Testing Scenarios

1. **No custom path configured** - Existing behavior unchanged, only HF cache used
2. **Custom path set, model exists there** - Loads from custom path
3. **Custom path set, model only in HF cache** - Falls back to HF cache
4. **Custom path set but directory doesn't exist** - Warning logged, HF-only mode
5. **Model in both locations** - Custom path takes precedence
6. **CLI overrides env var** - `--model-path` wins over `MLX_OMNI_MODEL_PATH`

### Test Path

Use `/Volumes/LLMS/models` for testing with the org/model structure.

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Custom path is a file, not directory | Error at startup with clear message |
| Model directory exists but no `config.json` | Skip, not treated as valid model |
| Permissions error reading custom path | Warning logged, continue with HF cache |
| Empty custom path directory | Valid, just returns empty list from that source |
| Symlinked model directories | Followed normally (standard filesystem behavior) |

### Logging

- Startup: Log configured model path (or "using HuggingFace cache only")
- Model load: Debug log which source the model was loaded from
- Fallback: Debug log when falling back from custom path to HF cache

## Out of Scope

- Automatic model downloading to custom path (users manage this manually)
- Model migration tools (copy from HF cache to custom path)
- Support for embeddings/images/TTS/STT models (can be added later with same pattern)
