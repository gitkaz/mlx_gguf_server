from typing import Generator, Dict, Any, List
import os
import uuid
import time
import json
import logging
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from transformers import PreTrainedTokenizer

from ...llm_model import LLMModel
from ...task.debug_info.debug_info import DebugInfo
from ...task.token_count.tokenizer_service import TokenizerService
from ...task_response import TaskResponse
from ...kv_cache.kv_cache_manager import KVCacheManager
from schemas import CompletionParams

from ...logger_config import setup_logger

log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)


class GenerationService:
    """
    A service that specializes in LLM text generation (completions_stream).
    Delegated from LLMModel, it performs the generation process.
    """

    def __init__(self, llm_model: LLMModel, params: Any, cache_manager: KVCacheManager, debug_info: DebugInfo):
        self.tokenizer_service = TokenizerService()
        self.kvcache_manager = cache_manager
        self.debug_info = debug_info

        if not isinstance(llm_model, LLMModel):
            raise TypeError("First argument must be LLMModel instance")
        
        self.model = llm_model.model
        self.model_capabilities = llm_model.capabilities
        self.tokenizer :PreTrainedTokenizer = llm_model.tokenizer
        self.model_type :str = llm_model.model_type
        self.model_name :str = llm_model.model_name
        self.default_gen_params :Dict[str, Any] = llm_model.default_gen_params
        self.params = params


    def generate_completion(self) -> Generator[Dict[str, Any], None, None]:

        def exception_response(e: Exception, source: str) -> Generator[Dict[str, Any], None, None]:
            error_msg = f"Error in GenerationService.generate_completion ({source}): {str(e)}"
            yield TaskResponse.create(500, {"error": error_msg}).to_dict()

        try:
            if self.model_type == "mlx":
                yield from self._generate_mlx()
            elif self.model_type == "llama-cpp":
                yield from self._generate_llama_cpp()
            else:
                yield TaskResponse.create(500, {"error": f"Unsupported model type: {self.model_type}"}).to_dict()
        except Exception as e:
            yield from exception_response(e, "main")


    def _generate_mlx(self) -> Generator[Dict[str, Any], None, None]:
        from mlx_lm.generate import stream_generate

        model = self.model
        model_name = self.model_name
        tokenizer = self.tokenizer

        # パラメータの統合
        gen_params = self._build_merged_params()
        logger.debug(f"mereged params: {gen_params=}")
        sampler = make_sampler(gen_params.temperature, top_p=gen_params.top_p)
        logits_processors = make_logits_processors(
            gen_params.logit_bias,
            gen_params.repetition_penalty,
            gen_params.repetition_context_size
        )

        request_id = str(uuid.uuid4())
        created_time = int(time.time())
        all_generated_tokens = []

        prompt = ""
        kv_cache = None
        kv_cache_metadata = {}

        try:
            # create prompt_tokens
            if gen_params.apply_chat_template:
                messages = gen_params.messages

                prompt = self.tokenizer_service.apply_chat_template(
                    tokenizer=tokenizer,
                    messages=messages,
                    add_generation_prompt=True,
                    tools=gen_params.tools,
                    enable_thinking=gen_params.enable_thinking,
                    model_capabilities=self.model_capabilities
                )
                prompt_tokens = tokenizer.encode(prompt)
            else:
                prompt = gen_params.prompt
                prompt_tokens = tokenizer.encode(prompt)

            # prompt_tokens based load kv cache
            if gen_params.use_kv_cache:
                kv_cache, start_index, kv_load_stats = self.kvcache_manager.load_kv_cache(model_name, prompt_tokens)

                # Cache hit by prompt_tokens based load_kv_cache
                if kv_cache is not None:
                    prompt_tokens = prompt_tokens[start_index:]

                # Cache hit failed by prompt_tokens based load_kv_cache
                else:
                    logger.debug(f"kv cache hit failed by prompt tokens based load_kv_cache.")

                    # If the request is /v1/chat/completions (messages), fallback to meessage_id based load_kv_cache
                    if gen_params.apply_chat_template: 
                        logger.debug(f"try to load kv cache based on message_id.")
                        kv_cache, message_index, kv_load_stats, tokens_in_metadata = self.kvcache_manager.load_kv_cache_by_message_id(model_name, messages)

                        # Cache hit by message_id based load_kv_cache
                        if kv_cache is not None:
                            trimmed_prompt = self.tokenizer_service.apply_chat_template(
                                tokenizer=tokenizer,
                                messages=messages[message_index:],
                                add_generation_prompt=True,
                                tools=gen_params.tools
                            )
                            trimmed_prompt_tokens = tokenizer.encode(trimmed_prompt)
                            
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"{tokenizer.decode(prompt_tokens)=}")
                                logger.debug(f"{tokenizer.decode(tokens_in_metadata)=}")
                                logger.debug(f"{tokenizer.decode(trimmed_prompt_tokens)=}")
                            prompt_tokens = trimmed_prompt_tokens

                # if use_kv_cache is enabled and kv_cache is not hit, create a new cache
                if kv_cache is None:
                    logger.info(f"kv cache hit failed. create a new kv cache.")
                    kv_cache, kv_load_stats = self.kvcache_manager.make_new_cache(model)

                self.debug_info.set(kv_cache = kv_load_stats)


            # generate text based on prompt_tokens and kv_cache (if params.use_kv_cache is True)
            start_time = time.perf_counter()
            is_first_token = True

            self.debug_info.set(
                stream_generate = {
                    "model": self.model_name, 
                    "prompt": self.tokenizer.decode(prompt_tokens)
                }
            )

            for item in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=gen_params.max_tokens,
                prompt_cache=kv_cache,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                text = item.text
                token = item.token
                all_generated_tokens.append(token)

                # check stop sequence
                if self._check_stop(gen_params.stop, all_generated_tokens, tokenizer):
                    del all_generated_tokens[-1]
                    break

                if gen_params.stream:
                    response = self._create_stream_chunk(
                        request_id, created_time, model_name, gen_params.apply_chat_template, text
                    )
                    yield response

                if is_first_token:
                    prompt_eval_time = time.perf_counter()
                    is_first_token = False

            # 非stream時の最終結果
            if not gen_params.stream:
                text = tokenizer.decode(all_generated_tokens)
                response = self._create_final_response(
                    request_id, created_time, model_name, gen_params.apply_chat_template, text
                )
                response["usage"] = self._calculate_usage(prompt, all_generated_tokens, tokenizer, start_time, prompt_eval_time)
                yield response

            # complete_textの有無に関わらず、usageを含む応答を返す
            generate_time = time.perf_counter()
            perf_timer = [start_time, prompt_eval_time, generate_time] if 'prompt_eval_time' in locals() else []

            response = self._create_usage_response(request_id, created_time, model_name)
            response["usage"] = self._calculate_usage(prompt, all_generated_tokens, tokenizer, *perf_timer)

            # KV Cacheの保存
            if gen_params.use_kv_cache and kv_cache is not None:
                should_save = (len(prompt_tokens) > gen_params.kv_cache_threshold)
                if should_save:
                    logger.debug(f"prompt_token lengh ({len(prompt_tokens)}) overed the threashold ({gen_params.kv_cache_threshold}). save kv cache.")

                    cached_tokens = tokenizer.encode(prompt) + all_generated_tokens
                    if cached_tokens[-1] in tokenizer.eos_token_ids:
                        cached_tokens = cached_tokens[:-1]

                    kv_cache_metadata["model_name"] = str(model_name)
                    kv_cache_metadata["chat_template"] = str(tokenizer.chat_template)
                    kv_cache_metadata["tokens"] = json.dumps(cached_tokens)
                    kv_cache_metadata["token_count"] = str(len(cached_tokens))
                    self.kvcache_manager.save_kv_cache(message_id=request_id, kv_cache=kv_cache, metadata=kv_cache_metadata)

                    response["usage"]["kv_cache"] = {
                        "cached_tokens": kv_cache_metadata["token_count"],
                        "stats": kv_load_stats
                    }

            yield response

            if gen_params.stream:
                yield {"stream_done": True}

        except Exception as e:
            yield from self._exception_response(e, "mlx")

    def _generate_llama_cpp(self, model: Any, params: Any) -> Generator[Dict[str, Any], None, None]:

        def get_llama_cpp_params(params) -> dict:
            return {
                'temperature'    : getattr(params, 'temperature', 1.0),
                'max_tokens'     : getattr(params, 'max_tokens', 4096),
                'stream'         : getattr(params, 'stream', False),
                'top_p'          : getattr(params, 'top_p', 0.95),
                'min_p'          : getattr(params, 'min_p', 0.05),
                'typical_p'      : getattr(params, 'typical_p', 1.0),
                'stop'           : getattr(params, 'stop', []),
                'frequency_penalty': getattr(params, 'frequency_penalty', 0.0),
                'presence_penalty' : getattr(params, 'presence_penalty', 0.0),
                'repeat_penalty' : getattr(params, 'repeat_penalty', 1.1),
                'top_k'          : getattr(params, 'top_k', 40),
                'seed'           : getattr(params, 'seed', None),
                'mirostat_mode'  : getattr(params, 'mirostat_mode', 2),
                'mirostat_tau'   : getattr(params, 'mirostat_tau', 5.0),
                'mirostat_eta'   : getattr(params, 'mirostat_eta', 0.1),
            }

        llama_cpp_params = get_llama_cpp_params(params)
        all_decoded_tokens = []

        try:
            if params.apply_chat_template:
                if params.stream:
                    for response in model.create_chat_completion(messages=params.messages, **llama_cpp_params):
                        all_decoded_tokens.append(response["choices"][0]["delta"].get("content", ""))
                        yield response
                else:
                    response = model.create_chat_completion(messages=params.messages, **llama_cpp_params)
                    yield response
            else:
                if params.stream:
                    for response in model.create_completion(prompt=params.prompt, **llama_cpp_params):
                        all_decoded_tokens.append(response["choices"][0]["text"])
                        yield response
                else:
                    response = model.create_completion(prompt=params.prompt, **llama_cpp_params)
                    yield response

            if params.complete_text:
                complete_text = "".join([t for t in all_decoded_tokens if isinstance(t, str)])
                usage = {"prompt_tokens": 0, "completion_tokens": len(all_decoded_tokens), "total_tokens": len(all_decoded_tokens)}
                yield {
                    "id": response.get("id"),
                    "object": response.get("object"),
                    "created": response.get("created"),
                    "model": response.get("model"),
                    "choices": [{"text": "", "complete_text": complete_text}],
                    "usage": usage
                }

            if params.stream:
                yield {"stream_done": True}

        except Exception as e:
            yield from self._exception_response(e, "llama-cpp")

    def _build_merged_params(self) -> CompletionParams:
        # Start with model defaults (if any)
        base_params = CompletionParams(**self.default_gen_params) if self.default_gen_params else CompletionParams()

        # Update only with explicitly provided request parameters
        return base_params.model_copy(
            update={k: v for k, v in self.params if k in self.params.model_fields_set}
        )

    def _check_stop(self, stop: List, tokens: List, tokenizer: Any) -> bool:
        if not stop:
            return False
        text = tokenizer.decode(tokens)
        return any(text.endswith(s) for s in stop)

    def _create_stream_chunk(self, rid: str, created: int, model: str, chat: bool, text: str):
        if chat:
            return {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"delta": {"content": text}}]
            }
        else:
            return {
                "id": rid,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"text": text}]
            }

    def _create_final_response(self, rid: str, created: int, model: str, chat: bool, text: str):
        if chat:
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"message": {"content": text}}]
            }
        else:
            return {
                "id": rid,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"text": text}]
            }

    def _create_usage_response(self, rid: str, created: int, model: str):
        return {
            "id": rid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"text": "", "complete_text": ""}]
        }

    def _calculate_usage(self, prompt: str, tokens: List, tokenizer: PreTrainedTokenizer, *perf_timer) -> Dict[str, Any]:
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokens)
        total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

        if len(perf_timer) == 3:
            usage["prompt_eval_time"] = perf_timer[1] - perf_timer[0]
            usage["generation_time"] = perf_timer[2] - perf_timer[1]

        return usage

    def _exception_response(self, e: Exception, source: str) -> Generator[Dict[str, Any], None, None]:
        error_msg = f"Error in GenerationService._generate_{source}: {str(e)}"
        yield TaskResponse.create(500, {"error": error_msg}).to_dict()