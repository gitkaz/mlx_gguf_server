# Automatic Model Unloading Feature

## Overview

The Automatic Model Unloading feature is designed to prevent memory exhaustion when loading multiple models.
When enabled, the mlx_gguf_server automatically unloads previously loaded models to make room for new model loading requests when system memory constraints are reached.

## Enabling the Feature

To enable automatic unloading, specify the maximum memory limit when starting the server:

```bash
python main.py --max-memory-gb 20
```

This sets a 20 GB memory limit for all loaded models combined. If `--max-memory-gb` argument not specified, this feature turnes off.


## Requirement memory calculation:
This feature calculates the memory requirements for a model based solely on the model's file size. (File size: approximately 20GB -> required memory: 20GB.) In actual use, the memory additionaly required to run the program and memory for the KV cache and so on.

## How It Works

### Core Mechanism

When a new model load request arrives:
1. The system calculates the memory required for the new model
2. It sums the memory usage of all currently loaded models
3. If the combined memory would exceed the configured limit:
   - The system identifies candidate models for unloading based on priority rules
   - It unloads candidates one by one until sufficient memory is available
   - If no candidates exist or unloading isn't enough, the load request fails

### Unloading Priority Rules

When multiple models qualify for unloading, they are prioritized in this order:
1. **Auto-unload flag**: Only models marked with `auto_unload=true` are considered
2. **Priority value**: Lower numerical priority values are unloaded first
3. **Memory size**: Larger models are preferred for unloading (more memory freed per unload)
4. **Last accessed time**: Older models are unloaded before recently used ones


## Usage Details

### Model Load Request Parameters

When loading models, you can control their behavior in the unloading system:

```json
{
  "llm_model_name": "your-model-name",
  "auto_unload": true,
  "priority": 0
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `auto_unload` | Whether this model can be automatically unloaded | `true` |
| `priority` | Unloading priority (lower = unloaded first) | `0` |
