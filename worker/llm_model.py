import json
import mlx.core as mx
import mlx_lm
import glob
import os
import time
import uuid
from llama_cpp import Llama
from typing import Generator, List, Dict, Any
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from .model_loader import ModelLoader
from .kv_cache_manager import load_kv_cache, save_kv_cache, clean_kv_cache
from .task_response import TaskResponse
from .logger_config import setup_logger

log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)

import gc


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


class LLMModel:
    def __init__(self):
        self.model_path: str = ""        
        self.model_name: str = ""
        self.model_type: str = ""
        self.model_cache_limit: int = 0
        self.model = None
        self.tokenizer = None
        self.default_gen_params: Dict[str, Any] = {}
        self.model_loader = ModelLoader()

    def load_model(self, params) -> TaskResponse:
        request_model_path = params.llm_model_path
        chat_format = params.chat_format
        logger.debug(f"start loading model: {request_model_path}.")
        start_time = time.time()
        try:
            loaded_model = self.model_loader.load(request_model_path, chat_format)

            self.model = loaded_model["model"]
            self.tokenizer = loaded_model["tokenizer"]
            self.model_type = loaded_model["type"]
            self.model_path = loaded_model["path"]
            self.model_name = os.path.basename(self.model_path)

            context_length  = loaded_model["context_length"]

        except Exception as e:
            logger.error(str(e))
            error_messsage = f"load failed: {request_model_path}. Reason={str(e)}"
            return TaskResponse.create(500, error_messsage)

        self.default_gen_params = {}
        for param_name in ["temperature", "max_tokens", "logit_bias", 
                          "repetition_penalty", "repetition_context_size", "top_p"]:
            value = getattr(params, param_name, None)
            if value is not None:  # Noneでなければ保存
                self.default_gen_params[param_name] = value


        load_time = time.time() - start_time
        self.model_path = request_model_path
        self.model_name = os.path.basename(request_model_path)

        logger.debug(f"loadded model: {self.model_name}. time: {load_time}s")
        model_info = {"model_name": self.model_name, 
                      "model_path":self.model_path, 
                      "model_type": self.model_type, 
                      "context_length": context_length, 
                      "defalt_params": self.default_gen_params,
                      "load_time":load_time
                      }
        logger.debug(f"{model_info=}")
        return TaskResponse.create(200, model_info)

    def get_mlx_params(self, params) -> dict:
        fallback_defaults = {
            'temperature': 1.0,
            'max_tokens': 4096,
            'logit_bias': None,
            'repetition_penalty': None,
            'repetition_context_size': 20,
            'top_p': 1.0,
            'tools': None,
        }

        # 1. モデルに設定されたカスタムデフォルト値
        result = self.default_gen_params.copy()
        # 2. カスタムデフォルトにない項目はフォールバック値で補完
        for k, v in fallback_defaults.items():
            if k not in result:
                result[k] = v
        # 3. リクエストパラメータで上書き（Noneでなければ）
        for k in result.keys():
            param_value = getattr(params, k, None)
            if param_value is not None:  # クライアントが明示的に送った値のみ採用
                result[k] = param_value

        return result

    def get_mlx_extension_params(self, params) -> dict:
        return {
            'stop'          : getattr(params, 'stop', []),
            'use_kv_cache'  : getattr(params, 'use_kv_cache', False),
        }


    def set_cache_liimt(self, params) -> TaskResponse:
        limit = int(params.cache_limit)
        try:
            if limit >= 0:
                logger.debug(f"set model cache limit {self.model_name=}, limit = {limit}")
                previous_limit = mx.metal.set_cache_limit(limit)
                self.model_cache_limit = limit
                message = f"change cache limit from {previous_limit} to {self.model_cache_limit}"
                logger.debug(message)
            return TaskResponse(200, message)
        except Exception as e:
            return TaskResponse(500, e)

    def get_cache_memory(self) -> TaskResponse:
        try:
            cache_memory_size = mx.metal.get_cache_memory()
            logger.debug(f"{cache_memory_size=}")
            return TaskResponse(200, f"{cache_memory_size=}")
        except Exception as e:
            return TaskResponse(500, e)
        
    def force_metal_clear_cache(self):
        logger.debug(f"mx.metal.get_cache_memory()={mx.metal.get_cache_memory()}")
        mx.metal.clear_cache()
        logger.debug(f"mx.metal.get_cache_memory()={mx.metal.get_cache_memory()}")

    def token_count(self, params) -> TaskResponse:
        try:
            if self.model_type == 'mlx':
                if params.messages != []:
                    tokenized_input = self.tokenizer.apply_chat_template(params.messages, tokenize=True, add_generation_prompt=True)
                else:
                    tokenized_input = self.tokenizer.tokenize(params.prompt)
            elif self.model_type == 'llama-cpp':
                if params.messages != []:
                    text = json.dumps(params.messages)                
                else:
                    text = params.prompt
                text = bytes(text, 'utf-8')
                tokenized_input= self.model.tokenize(text)

            token_length = len(tokenized_input)
            return TaskResponse(200, token_length)
        except Exception as e:
            return TaskResponse(500, e)

    def completions_stream(self, params) -> Generator[TaskResponse, None, None]:
        mlx_llama_generate = MLX_LLAMA_Generate(
            self.model, 
            self.tokenizer, 
            self.model_type, 
            self.model_name, 
            self.default_gen_params
        )

        logger.debug("start completions_stream")

        for response in mlx_llama_generate.generate_completion(params):

            if not isinstance(response, dict):
                error_message = f"error: generate_completion: response is not dict. {response=}"
                logger.error(error_message)
                response = error_message
                status = 500
            elif response.get("error"):
                status = 500
            else:
                status = 200
            yield TaskResponse.create(status, response)

        # logger.debug("completions_stream: Garbage Collect.")
        # del mlx_llama_generate 
        # gc.collect()
   

    def apply_chat_template(self, messages: List[dict], tools=None) -> str:
        chatml_instruct_template="{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '<|im_start|>system\n' + message['content'].rstrip() + '<|im_end|>\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'<|im_start|>user\n' + message['content'].rstrip() + '<|im_end|>\n'-}}{%- else -%}{{-'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'<|im_start|>assistant\n'-}}{%- endif -%}"

        try:
            chat_prompt = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
            logger.debug(f"{chat_prompt=}")
        except:
            logger.warn("apply chat template failed. try default format.")
            try:
                self.tokenizer.chat_template = self.tokenizer.default_chat_template
                chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                logger.debug(f"{chat_prompt=}")
            except:
                logger.warn("apply chat template failed. try fallback format.")
                chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chatml_instruct_template)
                logger.debug(f"{chat_prompt=}")
        return chat_prompt


class MLX_LLAMA_Generate(LLMModel):
    def __init__(self, model, tokenizer, model_type, model_name, parent_default_params: Dict[str, Any]):
        # super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.model_name = model_name
        self.default_gen_params = parent_default_params.copy()

    def generate_completion(self, params) -> Generator[dict, None, None]:
        def exception_message_in_generate(e: Exception, model_type: str) -> dict:
            error_message = f"Error in {model_type} generate_completion: {str(e)}"
            logger.error(error_message)
            response = {"error": error_message}
            return response

        def calculate_mlx_usage(self, prompt:str, all_tokens:List, perf_timer:List = []) -> dict:
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = len(all_tokens)
            total_tokens = prompt_tokens + completion_tokens
            if len(perf_timer) == 3:
                prompt_eval_time = perf_timer[1] - perf_timer[0]
                generation_time  = perf_timer[2] - perf_timer[1]
            else:
                prompt_eval_time = "not measured"
                generation_time  = "not measured"

            usage = {}
            usage = {"prompt_tokens": prompt_tokens, 
                     "completion_tokens":completion_tokens, 
                     "total_tokens": total_tokens,
                     "prompt_eval_time": prompt_eval_time,
                     "generation_time" : generation_time
                    }
            return usage

        def check_stop_keywords(stop: List, tokens: List):
            stop_sequence_matched: bool = False
            if stop:
                for stop_sequence in stop:
                    text = self.tokenizer.decode(tokens)
                    if text.endswith(stop_sequence):
                        logger.debug(f"stop keywords found. kewords={stop_sequence}")
                        return True
            return False

        if self.model_type == 'mlx':
            mlx_params = self.get_mlx_params(params)
            mlx_ext_params = self.get_mlx_extension_params(params)
            sampler = make_sampler(mlx_params["temperature"], top_p=mlx_params["top_p"])
            logits_processors = make_logits_processors(
                mlx_params["logit_bias"],
                mlx_params["repetition_penalty"], 
                mlx_params["repetition_context_size"]
            )

            request_id = str(uuid.uuid4())
            created_time = int(time.time())
            all_tokens = []
            kv_cache = None
            logger.debug(f"{mlx_params=}")

            if params.apply_chat_template:
                messages = params.messages

                # kv cache の生成
                if mlx_ext_params["use_kv_cache"]:
                    kv_cache, kv_cache_metadata, index, kv_load_stats = load_kv_cache(self.model, messages)
                    params.prompt = self.apply_chat_template(messages[index:], tools=mlx_params["tools"])
                else:
                    params.prompt = self.apply_chat_template(messages, tools=mlx_params["tools"])
                logger.debug(f"Chat Template applied {params.prompt=}")

            start_time = time.perf_counter()
            is_first_token = True

            try:
                for item in stream_generate(
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    prompt=params.prompt,
                    max_tokens=params.max_tokens,
                    prompt_cache=kv_cache,
                    sampler = sampler,
                    logits_processors = logits_processors,
                    ):
                    text = item.text
                    tokens = item.token
                    all_tokens.append(tokens)

                    if check_stop_keywords(params.stop, all_tokens) is True:
                        del all_tokens[-1]
                        break

                    if params.apply_chat_template:
                        if params.stream:
                            response = {"id": request_id,
                                        "object": "chat.completion.chunk", 
                                        "created": created_time, 
                                        "model": self.model_name, 
                                        "choices": [{"delta": {"content": text}}]
                                        }
                        else:
                            continue
                    else:
                        if params.stream:
                            response = {"id": request_id, 
                                        "object": "text_completion", 
                                        "created": created_time, 
                                        "model": self.model_name, 
                                        "choices": [{"text": text}]
                                        }
                        else:
                            continue
                    if is_first_token:
                        prompt_eval_time = time.perf_counter()
                        is_first_token = False
                    yield response

                if not params.stream:
                    if params.apply_chat_template:
                        text = self.tokenizer.decode(all_tokens)
                        response = {"id": request_id, 
                                    "object": "chat.completion", 
                                    "created": created_time, 
                                    "model": self.model_name, 
                                    "choices": [{"message": { "content": text}}]
                                    }
                        response["usage"] = calculate_mlx_usage(self, params.prompt, all_tokens)
                        yield response
                    else:
                        text = self.tokenizer.decode(all_tokens)
                        response = {"id": request_id, 
                                    "object": "text_completion", 
                                    "created": created_time, 
                                    "model": self.model_name, 
                                    "choices": [{"text": text}]
                                    }
                        response["usage"] = calculate_mlx_usage(self, params.prompt, all_tokens)
                        yield response

                if params.complete_text:
                    complete_text = self.tokenizer.decode(all_tokens)
                else:
                    complete_text = ""

                generate_time = time.perf_counter()
                perf_timer = [start_time, prompt_eval_time, generate_time]
                response = {"id": request_id, 
                            "object": "text_completion", 
                            "created": created_time, 
                            "model": self.model_name, 
                            "choices": [{"text": "", "complete_text": complete_text}]
                            }
                response["usage"] = calculate_mlx_usage(self, params.prompt, all_tokens, perf_timer)
                if kv_cache:
                    response["usage"]["kv_cache"] = {}
                    cached_token_count = int(response["usage"]["total_tokens"]) + int(kv_cache_metadata["token_count"])
                    kv_cache_metadata["model_name"]    = str(self.model_name)
                    kv_cache_metadata["chat_template"] = str(self.tokenizer.chat_template)
                    kv_cache_metadata["token_count"]   = str(cached_token_count)
                    save_kv_cache(message_id=request_id, kv_cache=kv_cache, metadata=kv_cache_metadata)
                    response["usage"]["kv_cache"]["cached_tokens"] = kv_cache_metadata["token_count"]
                    response["usage"]["kv_cache"]["stats"] = kv_load_stats
                yield response
    
            except Exception as e:
                logger.error(f"Error in generate_completion: {str(e)}, text: {text}, tokens: {tokens}")
                yield exception_message_in_generate(e, self.model_type)

        elif self.model_type == 'llama-cpp':
            llama_cpp_params = get_llama_cpp_params(params)
            logger.debug(f"{llama_cpp_params=}")
            try:
                if params.apply_chat_template:
                    if params.stream:
                        all_decoded_tokens = []
                        for response in self.model.create_chat_completion(
                            messages=params.messages,
                            **llama_cpp_params
                            ):
                            all_decoded_tokens.append(response["choices"][0]["delta"].get("content"))
                            yield response
                    else:
                        response = self.model.create_chat_completion(
                            messages=params.messages,
                            **llama_cpp_params
                            )
                        yield response

                else:
                    if params.stream:
                        all_decoded_tokens = []
                        for response in self.model.create_completion(
                            prompt=params.prompt,
                            **llama_cpp_params
                            ):
                            all_decoded_tokens.append(response["choices"][0]["text"])
                            yield response
                    else:
                        response = self.model.create_completion(
                            prompt=params.prompt,
                            **llama_cpp_params
                            )
                        yield response

                if params.complete_text:
                    all_str_tokens = [token for token in all_decoded_tokens if isinstance(token, str)]
                    complete_text = "".join(all_str_tokens)
                    response = {"id": response["id"], 
                                "object":response["object"], 
                                "created": response["created"], 
                                "model": response["model"],
                                "choices": [{"text": "", "complete_text": complete_text}]
                                }
                    response["usage"] = {"prompt_tokens":0, "completion_tokens":len(all_decoded_tokens), "total_tokens": len(all_decoded_tokens)}
                    yield response

            except Exception as e:
                yield exception_message_in_generate(e, self.model_type)

        if params.stream:
            response = {"stream_done": True}
            yield response
