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
from .tokenizer_service import TokenizerService
from .generation_service import GenerationService
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
        self.tokenizer_service = TokenizerService()
        self.generator = GenerationService(self.tokenizer_service)

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

    def token_count(self, params) -> TaskResponse:
        try:
            # TokenizerService に委譲
            token_length = self.tokenizer_service.count_tokens(
                model_type=self.model_type,
                tokenizer=self.tokenizer,
                prompt=params.prompt,
                messages=params.messages
            )
            return TaskResponse(200, token_length)
        except Exception as e:
            return TaskResponse(500, str(e))

    def completions_stream(self, params) -> Generator[TaskResponse, None, None]:
        try:
            for response in self.generator.generate_completion(
                model=self.model,
                tokenizer=self.tokenizer,
                model_type=self.model_type,
                model_name=self.model_name,
                default_gen_params=self.default_gen_params,
                params=params
            ):
                yield TaskResponse.create(200, response)
        except Exception as e:
            yield TaskResponse.create(500, {"error": str(e)})

def apply_chat_template(self, messages: List[dict], tools=None) -> str:
    try:
        # TokenizerService に委譲
        return self.tokenizer_service.apply_chat_template(
            tokenizer=self.tokenizer,
            messages=messages,
            tools=tools,
            add_generation_prompt=True
        )
    except Exception as e:
        raise RuntimeError(f"Chat template failed: {str(e)}")
