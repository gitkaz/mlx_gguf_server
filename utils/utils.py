import os
import math
import json

def get_unload_candidates(loaded_models: dict):
    return sorted(
        [
            (model_name, entry) 
            for model_name, entry in loaded_models.items()
            if isinstance(entry, dict) and entry["auto_unload"]
        ],
        key=lambda x: (x[1]["priority"], -x[1]["memory_usage"], x[1]["last_accessed"])
    )


def get_model_size(model_path: str):
    model_size = 0
    if os.path.isfile(model_path):
        model_size = os.path.getsize(model_path)

    elif os.path.isdir(model_path):
        for file in os.scandir(model_path):
            if os.path.isfile(file):
                model_size += os.path.getsize(file)

    model_size_in_gb = model_size / (1024 * 1024 * 1024)
    model_size_in_gb = math.floor(model_size_in_gb)
    return model_size_in_gb


def create_model_list():
    models ={}
    for f in os.listdir("models"):
        if os.path.isdir(os.path.join("models", f)) or f.endswith(".gguf"):
            model_name = f
            model_path = os.path.join('models', model_name)
            model_size = get_model_size(model_path)
            models[model_name] = {'path': model_path, 'size': model_size }

    return models


def create_adapter_list():
    """
    Scan the repository's `adapters/` directory and return a mapping
    of {adapter_name: {path}}.
    Adapters may be a single file (for example, .safetensors) or a directory.
    """
    adapters = {}
    adapters_dir = "adapters"
    if not os.path.exists(adapters_dir):
        return adapters

    for f in os.listdir(adapters_dir):
        full = os.path.join(adapters_dir, f)
        if os.path.isdir(full) or (os.path.isfile(full) and f.endswith(".safetensors")):
            adapter_name = f
            adapter_path = full
            adapters[adapter_name] = {"path": adapter_path}

    return adapters

def _calculate_rope_context_length(rope_config: dict) -> int | None:
    """
    Calculate context length from rope_scaling or rope_parameters config.

    Args:
        rope_config: Dictionary containing rope scaling parameters

    Returns:
        int: Calculated context length, or None if parameters are missing
    """
    if not rope_config:
        return None

    factor = rope_config.get("factor")
    original_max = rope_config.get("original_max_position_embeddings")

    if factor and original_max:
        return int(factor * original_max)

    return None

def get_mlx_context_length(model_path: str) -> int:
    """
    Get context length from MLX model's config.json.
    Handles rope_scaling, rope_parameters, and various config structures.

    Args:
        model_path: Path to the MLX model directory

    Returns:
        int: Context length (max_position_embeddings)

    Raises:
        RuntimeError: If config cannot be read or parsed
    """
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        max_position_embeddings = None

        # Check for text_config first (multimodal models)
        if config.get("text_config"):
            text_config = config["text_config"]

            # Check for rope_parameters inside text_config
            if text_config.get("rope_parameters"):
                max_position_embeddings = _calculate_rope_context_length(text_config["rope_parameters"])

            # Check for rope_scaling inside text_config
            if max_position_embeddings is None and text_config.get("rope_scaling"):
                max_position_embeddings = _calculate_rope_context_length(text_config["rope_scaling"])

            # Fall back to direct max_position_embeddings in text_config
            if max_position_embeddings is None:
                max_position_embeddings = text_config.get("max_position_embeddings")

        # If no text_config, check root level (text-only models)
        if max_position_embeddings is None:
            if config.get("rope_scaling"):
                max_position_embeddings = _calculate_rope_context_length(config["rope_scaling"])

            if max_position_embeddings is None:
                max_position_embeddings = config.get("max_position_embeddings")

        if max_position_embeddings is None:
            raise RuntimeError(f"Could not find max_position_embeddings in {config_path}")

        return max_position_embeddings

    except Exception as e:
        raise RuntimeError(f"Failed to get max_position_embeddings from {config_path}") from e

def get_mlx_model_config(model_path: str) -> dict:
    """
    Read and parse config.json from an MLX model directory.

    Args:
        model_path: Path to the MLX model directory

    Returns:
        dict with configuration details (context_length, hidden_size, etc.)
        May contain 'config_error' key if reading failed
    """
    result = {}
    config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_path):
        result["config_error"] = "config.json not found"
        return result

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Use the shared context length function
        result["context_length"] = get_mlx_context_length(model_path)
        result["hidden_size"] = config.get("hidden_size")
        result["num_layers"] = config.get("num_hidden_layers")
        result["vocab_size"] = config.get("vocab_size")

    except Exception as e:
        result["config_error"] = str(e)

    return result


def get_model_details(model_name: str) -> dict:
    """
    Get model information without loading the model.
    Works for both MLX directories and GGUF files.

    Args:
        model_name: Name of the model (directory name or .gguf filename)

    Returns:
        dict with model information

    Raises:
        ValueError: If model not found
    """
    models = create_model_list()

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found")

    model_path = models[model_name]['path']
    is_gguf = model_path.lower().endswith('.gguf')

    result = {
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "llama-cpp" if is_gguf else "mlx",
        "file_size_gb": models[model_name]['size'],
        "is_loaded": False,  # Will be updated by caller if needed
    }

    # For MLX models, read config.json for additional details
    if not is_gguf:
        config_details = get_mlx_model_config(model_path)
        result.update(config_details)

    return result