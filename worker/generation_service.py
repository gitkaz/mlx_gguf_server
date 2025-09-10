from typing import Generator, Dict, Any, List
import mlx.core as mx
import uuid
import time
import json
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from transformers import PreTrainedTokenizer
from .kv_cache_manager import load_kv_cache, save_kv_cache
from .tokenizer_service import TokenizerService
from .llm_model import LLMModel

from .task_response import TaskResponse

from .logger_config import setup_logger
logger = setup_logger(__name__, level="DEBUG")


class GenerationService:
    """
    LLMのテキスト生成 (completions_stream) を専門に行うサービス。
    LLMModelから委譲されて、生成処理を実行する。
    """

    def __init__(self):
        self.tokenizer_service = TokenizerService()

    def generate_completion(self, llm_model: LLMModel, params :Any) -> Generator[Dict[str, Any], None, None]:
        """
        テキスト生成を実行。stream対応。
        """

        if not isinstance(llm_model, LLMModel):
            raise TypeError("First argument must be LLMModel instance")

        model = llm_model.model
        tokenizer = llm_model.tokenizer
        model_type = llm_model.model_type
        model_name = llm_model.model_name
        default_gen_params = llm_model.default_gen_params



        def exception_response(e: Exception, source: str) -> Generator[Dict[str, Any], None, None]:
            error_msg = f"Error in GenerationService.generate_completion ({source}): {str(e)}"
            yield TaskResponse.create(500, {"error": error_msg}).to_dict()

        try:
            if model_type == "mlx":
                yield from self._generate_mlx(
                    model, tokenizer, model_name, default_gen_params, params
                )
            elif model_type == "llama-cpp":
                yield from self._generate_llama_cpp(model, params)
            else:
                yield TaskResponse.create(500, {"error": f"Unsupported model type: {model_type}"}).to_dict()
        except Exception as e:
            yield from exception_response(e, "main")

    def _generate_mlx(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        default_gen_params: Dict[str, Any],
        params: Any
    ) -> Generator[Dict[str, Any], None, None]:
        from mlx_lm.generate import stream_generate

        # パラメータの統合
        gen_params = self._build_mlx_params(default_gen_params, params)
        sampler = make_sampler(gen_params["temperature"], top_p=gen_params["top_p"])
        logits_processors = make_logits_processors(
            gen_params["logit_bias"],
            gen_params["repetition_penalty"],
            gen_params["repetition_context_size"]
        )

        request_id = str(uuid.uuid4())
        created_time = int(time.time())
        all_generated_tokens = []

        prompt = ""
        kv_cache = None
        kv_cache_metadata = {}
        kv_load_stats = {}
        index = 0

        try:
            if params.apply_chat_template:
                messages = params.messages
                prompt = self.tokenizer_service.apply_chat_template(
                    tokenizer=tokenizer,
                    messages=messages[index:],
                    add_generation_prompt=True,
                    tools=gen_params["tools"]
                )
                prompt_tokens = tokenizer.encode(prompt)

                # KV Cacheの利用（use_kv_cacheがTrueの場合）
                if params.use_kv_cache:
                    kv_cache, kv_cache_metadata, start_index, kv_load_stats = load_kv_cache(model, prompt_tokens)
                    prompt_tokens = prompt_tokens[start_index:]
            else:
                prompt = params.prompt
                prompt_tokens = tokenizer.encode(prompt)


            # 生成処理
            start_time = time.perf_counter()
            is_first_token = True

            for item in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=params.max_tokens,
                prompt_cache=kv_cache,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                text = item.text
                token = item.token
                all_generated_tokens.append(token)

                # stop sequenceチェック
                if self._check_stop(params.stop, all_generated_tokens, tokenizer):
                    del all_generated_tokens[-1]
                    break

                if params.stream:
                    response = self._create_stream_chunk(
                        request_id, created_time, model_name, params.apply_chat_template, text
                    )
                    yield response

                if is_first_token:
                    prompt_eval_time = time.perf_counter()
                    is_first_token = False

            # 非stream時の最終結果
            if not params.stream:
                text = tokenizer.decode(all_generated_tokens)
                response = self._create_final_response(
                    request_id, created_time, model_name, params.apply_chat_template, text
                )
                response["usage"] = self._calculate_usage(prompt, all_generated_tokens, tokenizer, start_time, prompt_eval_time)
                yield response

            # complete_textの有無に関わらず、usageを含む応答を返す
            generate_time = time.perf_counter()
            perf_timer = [start_time, prompt_eval_time, generate_time] if 'prompt_eval_time' in locals() else []

            response = self._create_usage_response(request_id, created_time, model_name)
            response["usage"] = self._calculate_usage(prompt, all_generated_tokens, tokenizer, *perf_timer)

            # KV Cacheの保存
            if params.use_kv_cache and kv_cache is not None:
                cached_tokens = tokenizer.encode(prompt) + all_generated_tokens
                if cached_tokens[-1] in tokenizer.eos_token_ids:
                    cached_tokens = cached_tokens[:-1]
                logger.debug(f"{tokenizer.decode(cached_tokens)=}")
                cached_token_count = response["usage"]["total_tokens"] + int(kv_cache_metadata.get("token_count", 0))
                kv_cache_metadata["model_name"] = str(model_name)
                kv_cache_metadata["chat_template"] = str(tokenizer.chat_template)
                kv_cache_metadata["tokens"] = json.dumps(cached_tokens)
                kv_cache_metadata["token_count"] = str(cached_token_count)
                save_kv_cache(message_id=request_id, kv_cache=kv_cache, metadata=kv_cache_metadata)

                response["usage"]["kv_cache"] = {
                    "cached_tokens": kv_cache_metadata["token_count"],
                    "stats": kv_load_stats
                }

            yield response

            if params.stream:
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

    def _build_mlx_params(self, default: Dict[str, Any], params: Any) -> Dict[str, Any]:
        fallbacks = {
            'temperature': 1.0,
            'max_tokens': 4096,
            'logit_bias': None,
            'repetition_penalty': None,
            'repetition_context_size': 20,
            'top_p': 1.0,
            'tools': None,
        }
        result = {**default}
        for k, v in fallbacks.items():
            if k not in result:
                result[k] = v
        for k in result.keys():
            val = getattr(params, k, None)
            if val is not None:
                result[k] = val
        return result

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