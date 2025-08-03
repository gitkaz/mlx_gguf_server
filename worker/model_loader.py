import mlx_lm
from llama_cpp import Llama

from typing import Dict, Any
import os
import json

class ModelLoader:
    def load(self, model_path: str, chat_format: str = None):
        if self._is_gguf_model(model_path):
            return self._load_llama_cpp(model_path, chat_format)
        else:
            return self._load_mlx(model_path)

    def _is_gguf_model(self, model_path: str) -> bool:
        """パスがGGUF形式かどうかを判定"""
        if os.path.isfile(model_path):
            return model_path.lower().endswith(".gguf")
        return False

    def _load_mlx(self, model_path: str) -> Dict[str, Any]:
        try:
            model, tokenizer = mlx_lm.load(model_path, tokenizer_config={"trust_remote_code": None})
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