import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer
from typing import Optional, Union, List, Dict, Generator, Optional, Tuple, Union

from .kv_cache_manager import KVCacheManager
from mlx_lm.utils import apply_repetition_penalty, generate_step
from mlx_lm.sample_utils import top_p_sampling, min_p_sampling, categorical_sampling
from mlx_lm.models.base import KVCache, RotatingKVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

def generate_stream(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
    stream: bool = False,
    stop: List = [],
    kv_cache_session_id = None,
):
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    tokens = []

    stop_sequence_matched: bool = False

    # 関数を選択し、適切な引数で呼び出すヘルパー関数
    def call_generate_function(kv_cache_session_id, **kwargs):
        if kv_cache_session_id is not None:
            return ext_generate_step(kv_cache_session_id=kv_cache_session_id, **kwargs)
        else:
            kwargs.pop('kv_cache_session_id', None)
            return generate_step(**kwargs)

    for (token, prob), n in zip(
        call_generate_function(
            kv_cache_session_id=kv_cache_session_id,
            prompt=prompt_tokens,
            model=model,
            temp=temp,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            top_p=top_p,
        ),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)
        tokens.append(token)

        if stop:
            detokenizer.finalize()
            for stop_sequence in stop:
                if detokenizer.text.endswith(stop_sequence):
                    stop_sequence_matched = True
            if stop_sequence_matched:
                break

        if stream is False:
            continue

        detokenizer.finalize()
        response = (detokenizer.last_segment, [token])
        yield response

    if stream is False:

        detokenizer.finalize()
        response = (detokenizer.text, tokens)
        yield response


def ext_generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    logit_bias: Optional[Dict[int, float]] = None,
    prefill_step_size: int = 512,
    max_kv_size: Optional[int] = None,
    kv_cache_session_id: Optional[int] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        logit_bias (dictionary, optional): Additive logit bias.
        prefill_step_size (int): Step size for processing the prompt.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
        logprobs = logits - mx.logsumexp(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temp)
            elif min_p != 0.0:
                token = min_p_sampling(logits, min_p, min_tokens_to_keep, temp)
            else:
                token = categorical_sampling(logits, temp)

        return token, logprobs

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = prompt

    if KVCacheManager.has_cache(kv_cache_session_id):
        cache = KVCacheManager.get_cache(kv_cache_session_id)
    else: 
        if hasattr(model, "make_cache"):
            cache = model.make_cache()
        else:
            kv_heads = (
                [model.n_kv_heads] * len(model.layers)
                if isinstance(model.n_kv_heads, int)
                else model.n_kv_heads
            )
            if max_kv_size is not None:
                cache = [
                    RotatingKVCache(model.head_dim, n, max_size=max_kv_size, keep=4)
                    for n in kv_heads
                ]
            else:
                cache = [KVCache(model.head_dim, n) for n in kv_heads]

    repetition_context = prompt.tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    def _step(y):
        nonlocal repetition_context
        logits = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, logprobs = sample(logits)
            repetition_context.append(y.item())
        else:
            y, logprobs = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        return y, logprobs.squeeze(0)

    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[prefill_step_size:]

    y, logprobs = _step(y)

    mx.async_eval(y)
    while True:
        next_y, next_logprobs = _step(y)
        mx.async_eval(next_y)
        KVCacheManager.set_cache(kv_cache_session_id, cache) # yield 直前に session_cache を更新する
        yield y.item(), logprobs
        y, logprobs = next_y, next_logprobs

