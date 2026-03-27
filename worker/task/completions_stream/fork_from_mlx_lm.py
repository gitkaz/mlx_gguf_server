from typing import Any, List, Optional
import os
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import cache
import functools

from ...logger_config import setup_logger

log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)

def prefill_kv_cache(
    prompt: mx.array,
    model: nn.Module,
    *,
    prompt_cache: Optional[Any] = None,
    max_kv_size: Optional[int] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    input_embeddings: Optional[mx.array] = None,
) -> List[Any]:
    """
    This function was forked from mlx_lm/generate.py::generate_step function.

    Process prompt tokens through the model and update KV cache.

    This function only performs prompt processing (prefill), NOT generation.
    The purposse of this function is update KV Cache.

    Args:
        prompt (mx.array): Input prompt tokens.
        model (nn.Module): The model to use for processing.
        prompt_cache (List[Any], optional): Existing KV cache to update.
            If None, a new cache is created.
        max_kv_size (int, optional): Maximum KV cache size.
        prefill_step_size (int): Chunk size for processing prompt.
        kv_bits (int, optional): KV cache quantization bits.
        kv_group_size (int): KV cache quantization group size.
        quantized_kv_start (int): Step to start KV quantization.
        input_embeddings (mx.array, optional): Input embeddings instead of tokens.

    Returns:
        List[Any]: The updated KV cache.
    """

    if len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    # Create or use existing cache
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    # Setup quantization
    def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
        if kv_bits is None:
            return
        for e, c in enumerate(prompt_cache):
            if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
                prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    # Define model call function
    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    # Get generation stream (same as generate_step)
    generation_stream = mx.new_stream(mx.default_device())

    # Process prompt tokens in chunks
    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0

        # Process in chunks (prefill)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            mx.clear_cache()

        # Process final token
        _model_call(
            input_tokens=prompt[None],
            input_embeddings=(
                input_embeddings[None] if input_embeddings is not None else None
            ),
        )

        quantize_cache_fn(prompt_cache)
        mx.eval([c.state for c in prompt_cache])

    return prompt_cache