import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer
from typing import Optional, Union, List, Dict, Generator, Tuple

from .kv_cache_manager import KVCacheManager
from mlx_lm.utils import apply_repetition_penalty, generate_step
from mlx_lm.sample_utils import top_p_sampling, min_p_sampling, categorical_sampling
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .logger_config import setup_logger
logger = setup_logger(__name__, level="DEBUG")


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
    从模型生成文本。

    参数:
       model (nn.Module): 语言模型。
       tokenizer (PreTrainedTokenizer): 分词器或包装器。
       prompt (str): 文本提示。
       temp (float): 采样温度（默认 0）。
       max_tokens (int): 最多生成的 token 数（默认 100）。
       repetition_penalty (float, optional): 重复 token 的惩罚因子。
       repetition_context_size (int, optional): 用于重复惩罚的上下文 token 数量。
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    tokens = []

    stop_sequence_matched: bool = False

    # 选择生成函数并以适当参数调用的辅助函数
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

        try:
            detokenizer.finalize()
            response = (detokenizer.last_segment, [token])
        except Exception as e:
            logger.error(f"Error in generate_stream: {e}")
            response = ("?", [token])

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
    基于给定 prompt 从模型生成 token id 的生成器。

    参数:
        prompt (mx.array): 输入 prompt。
        model (nn.Module): 用于生成的模型。
        temp (float): 采样温度，若为 0 则使用 argmax（默认 0）。
        repetition_penalty (float, optional): 重复 token 的惩罚系数。
        repetition_context_size (int, optional): 用于重复惩罚的上下文 token 数量（默认 20）。
        top_p (float, optional): top-p 采样阈值。
        min_p (float, optional): 最小概率阈值（相对于最高概率 token 缩放）。
        min_tokens_to_keep (int, optional): 在 min_p 过滤下仍需保留的最小 token 数量。
        logit_bias (dict, optional): 加到 logits 上的偏置。
        prefill_step_size (int): 处理 prompt 时一次 prefill 的步长。
        max_kv_size (int, optional): key-value 缓存的最大大小。

    返回:
        生成 (token, logprobs) 的生成器。
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
        KVCacheManager.set_cache(kv_cache_session_id, cache)  # 在 yield 之前更新 session_cache
        yield y.item(), logprobs
        y, logprobs = next_y, next_logprobs

