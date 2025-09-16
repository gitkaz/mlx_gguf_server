import mlx_lm
from llama_cpp import Llama
from typing import Dict, Any, Optional
import os
import time
import json

from .task_response import TaskResponse
from .llm_model import LLMModel
from schemas import ModelLoadParams

from .logger_config import setup_logger
log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)


class ModelLoader:
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
                loaded_model = self._load_mlx(request_model_path, params.adapter_path)

            # 2. LLMModelインスタンスの設定
            self._configure_llm_model(llm_model, loaded_model, request_model_path)

            # 3. デフォルト生成パラメータの上書き
            self._set_default_generation_params(llm_model, params)

            # 4. ロード時間の計算とログ
            load_time = time.time() - start_time
            logger.debug(f"loaded model: {llm_model.model_name}. time: {load_time}s")

            # 5. レスポンス用データの作成
            model_info = self._create_model_info(llm_model, load_time)

            return TaskResponse.create(200, model_info)

        except Exception as e:
            return self._handle_load_error(request_model_path, e)

    def _is_gguf_model(self, model_path: str) -> bool:
        """パスがGGUF形式かどうかを判定"""
        if os.path.isfile(model_path):
            return model_path.lower().endswith(".gguf")
        return False

    def _load_mlx(self, model_path: str, adapter_path: Optional[str]) -> Dict[str, Any]:
        try:
            model, tokenizer = mlx_lm.load(model_path, tokenizer_config={"trust_remote_code": None}, adapter_path=adapter_path)
            context_length = self._get_mlx_context_length(model_path)

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
            "repetition_penalty", "repetition_context_size", "top_p", "use_kv_cache"
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

    def _get_mlx_context_length(self, model_path: str) -> int:
        """MLXモデルの context_length を config.json から取得"""
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            if config.get("rope_scaling") and \
            config["rope_scaling"].get("factor") and \
            config["rope_scaling"].get("original_max_position_embeddings"):
                factor = config["rope_scaling"].get("factor")
                original_max_position_embeddings = config["rope_scaling"].get("original_max_position_embeddings")
                max_position_embeddings = int(factor * original_max_position_embeddings)
                
            elif config.get("text_config"):
                max_position_embeddings = config["text_config"].get("max_position_embeddings")
            else:
                max_position_embeddings = config.get("max_position_embeddings")
            return max_position_embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to get max_position_embeddings from {config_path}") from e
        

    def _handle_load_error(self, model_path: str, error: Exception) -> TaskResponse:
        """ロードエラーを処理"""
        error_message = f"Model load failed for {model_path}: {str(error)}"
        logger.error(error_message, exc_info=True)

        # 特定のエラー型に応じた処理
        if isinstance(error, FileNotFoundError):
            return TaskResponse.create(404, "Model file not found")

        return TaskResponse.create(500, error_message)