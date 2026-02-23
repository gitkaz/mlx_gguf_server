import mlx_lm
from llama_cpp import Llama
from typing import Dict, Any, Optional
import os
import time
import json


from ...task_response import TaskResponse
from ...llm_model import LLMModel
from schemas import ModelLoadParams
from utils.utils import get_mlx_context_length

from ...logger_config import setup_logger
log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)


class ModelLoader:
    def __init__(self):
        self.model_capabilities = self._load_capabilities()

    def _load_capabilities(self) -> dict:
        """Load model capabilities from environment variable"""
        import os
        import json

        default_config = {
            "models": {},
            "defaults": {
                "supports_thinking_toggle": False,
                "thinking_variable": None,
                "default_thinking": True
            }
        }

        try:
            capabilities_json = os.environ.get("MODEL_CAPABILITIES")
            if capabilities_json:
                return json.loads(capabilities_json)
        except Exception as e:
            logger.warning(f"Failed to parse MODEL_CAPABILITIES: {e}")

        return default_config

    def get_model_capabilities(self, model_name: str) -> dict:
        """Get capabilities for a specific model"""
        models = self.model_capabilities.get("models", {})
        defaults = self.model_capabilities.get("defaults", {})

        # Exact match
        if model_name in models:
            return {**defaults, **models[model_name]}

        # Partial match (for version variants)
        for pattern, caps in models.items():
            if pattern in model_name:
                return {**defaults, **caps}

        return defaults

    def load(self, llm_model: LLMModel, params: ModelLoadParams) -> TaskResponse:
        """
        LLMModelインスタンスを直接操作してモデルをロード

        Args:
            llm_model: 操作対象のLLMModelインスタンス
            params: モデルロードパラメータ

        Returns:
            TaskResponse: ロード結果
        """

        if not isinstance(llm_model, LLMModel):
            raise TypeError("First argument must be LLMModel instance")

        request_model_path = params.llm_model_path
        logger.debug(f"start loading model: {request_model_path}.")
        start_time = time.time()

        try:
            # 1. 実際のモデルロード処理
            if self._is_gguf_model(request_model_path):
                loaded_model = self._load_llama_cpp(request_model_path, params.chat_format)
            else:
                loaded_model = self._load_mlx(request_model_path, params.adapter_path, trust_remote_code=params.trust_remote_code)

            # 2. LLMModelインスタンスの設定
            self._configure_llm_model(llm_model, loaded_model, request_model_path)

            # 3. デフォルト生成パラメータの上書き
            self._set_default_generation_params(llm_model, params)

            # 4. ロード時間の計算とログ
            load_time = time.time() - start_time
            logger.debug(f"loaded model: {llm_model.model_name}. time: {load_time}s")

            # 5. レスポンス用データの作成
            model_info = self._create_model_info(llm_model, load_time)

            # 6. Set model capabilities
            llm_model.capabilities = self.get_model_capabilities(llm_model.model_name)
            logger.debug(f"Model capabilities set: {llm_model.capabilities}")

            return TaskResponse.create(200, model_info)

        except Exception as e:
            return self._handle_load_error(request_model_path, e)

    def _is_gguf_model(self, model_path: str) -> bool:
        """パスがGGUF形式かどうかを判定"""
        if os.path.isfile(model_path):
            return model_path.lower().endswith(".gguf")
        return False

    def _load_mlx(self, model_path: str, adapter_path: Optional[str], trust_remote_code: Optional[bool] = None) -> Dict[str, Any]:

        use_trust = trust_remote_code if trust_remote_code is not None else False
        try:
            model, tokenizer = mlx_lm.load(model_path, tokenizer_config={"trust_remote_code": use_trust}, adapter_path=adapter_path, lazy=True)
            context_length = get_mlx_context_length(model_path)

            return {
                "model": model,
                "tokenizer": tokenizer,
                "type": "mlx",
                "path": model_path,
                "context_length": context_length,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load mlx model: {model_path}") from e

    def _load_llama_cpp(self, model_path: str, chat_format: str = None) -> Dict[str, Any]:
        try:
            model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Apple Silicon で最適化
                n_ctx=0,  # n_ctx=0 は自動で最大長を設定
                chat_format=chat_format,
                verbose=False,  # 詳細ログは別で制御
            )
            context_length = model.n_ctx()

            return {
                "model": model,
                "tokenizer": None,
                "type": "llama-cpp",
                "path": model_path,
                "context_length": context_length,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load llama.cpp model: {model_path}") from e

    def _configure_llm_model(self, llm_model, loaded_model: Dict[str, Any], model_path: str) -> None:
        """LLMModelインスタンスの基本プロパティを設定"""
        llm_model.model = loaded_model["model"]
        llm_model.tokenizer = loaded_model["tokenizer"]
        llm_model.model_type = loaded_model["type"]
        llm_model.model_path = loaded_model["path"]
        llm_model.model_name = os.path.basename(model_path)
        llm_model.context_length = loaded_model["context_length"]

        logger.debug(f"Model configured: {llm_model.model_name} ({llm_model.model_type})")


    def _set_default_generation_params(self, llm_model, params: ModelLoadParams) -> None:
        """LLMModelにデフォルト生成パラメータを設定"""
        llm_model.default_gen_params = {}
        relevant_params = [
            "temperature", "max_tokens", "logit_bias",
            "repetition_penalty", "repetition_context_size", "top_p", "use_kv_cache", "kv_cache_threshold"
        ]

        for param_name in relevant_params:
            value = getattr(params, param_name, None)
            if value is not None:
                llm_model.default_gen_params[param_name] = value

        logger.debug(f"Default generation params set: {llm_model.default_gen_params}")

    def _create_model_info(self, llm_model, load_time: float) -> Dict[str, Any]:
        """ロード結果の情報を含む辞書を作成"""
        model_info = {
            "model_name": llm_model.model_name,
            "model_path": llm_model.model_path,
            "model_type": llm_model.model_type,
            "context_length": llm_model.context_length,
            "default_params": llm_model.default_gen_params,
            "load_time": round(load_time, 2)
        }

        logger.debug(f"Model info created: {model_info}")
        return model_info        

    def _handle_load_error(self, model_path: str, error: Exception) -> TaskResponse:
        """ロードエラーを処理"""
        error_message = f"Model load failed for {model_path}: {str(error)}"
        logger.error(error_message, exc_info=True)

        # 特定のエラー型に応じた処理
        if isinstance(error, FileNotFoundError):
            return TaskResponse.create(404, "Model file not found")

        return TaskResponse.create(500, error_message)