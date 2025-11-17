# Automatic Model Loading Feature

## Overview

The Automatic Model Loading feature enables mlx_gguf_server to automatically load models when they're requested through API calls, rather than requiring manual pre-loading. This idea was inspired by LM Studio's "JIT (Just-in-time) Loading," which allows applications to use models without requiring explicit loading steps. I have deep respect and gratitude for all the developers at LM Studio.

This feature completes the memory management ecosystem when used alongside the Auto-Unload functionality, creating an intelligent model lifecycle management system.

## How It Works

When Auto-Load is enabled, the server automatically handles model loading in the background:

1. When an API request arrives for `/v1/completions` or `/v1/chat/completions` with a `model` parameter
2. The server checks if the requested model is already loaded
3. If not loaded, it automatically loads the model using sensible defaults
4. Once loaded, the request is processed normally
5. The model remains available for subsequent requests until Auto-Unload potentially removes it

## Enabling the Feature

To enable Auto-Load, you must:

1. First enable Auto-Unload by specifying `--max-memory-gb` (required)
2. Add the `--enable-auto-load` flag when starting the server

```bash
python main.py --max-memory-gb XX(num) --enable-auto-load
```

**Important Requirements:**
- Auto-Load **cannot** be enabled without Auto-Unload
- The `--max-memory-gb` parameter must be specified (Auto-Unload requirement)
- Auto-Load is **disabled by default** for safety

## Notes

### Default Parameters for Auto-Loaded Models

When a model is auto-loaded, it's initialized with these default parameters:

| Parameter | Value |
|-----------|-------|
| `auto_unload` | `true` |
| `priority` | `0` |
| `use_kv_cache` | `true` |

### Model ID Assignment

Auto-loaded models are assigned IDs in the range of 100-199 to distinguish them from manually loaded models (typically 0-99). 