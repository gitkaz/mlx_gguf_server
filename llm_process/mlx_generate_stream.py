import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer
from typing import Optional, Union, List

from mlx_lm.utils import generate_step
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
    for (token, prob), n in zip(
        generate_step(
            prompt_tokens,
            model,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
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
