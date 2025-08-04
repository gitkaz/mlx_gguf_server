from typing import Dict, Any, List
from transformers import PreTrainedTokenizer
import os
import json

from .logger_config import setup_logger

log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)


class TokenizerService:
    """
    トークナイザに関する操作を専門に行うサービス
    - チャットテンプレートの適用
    - トークン数のカウント
    """

    def apply_chat_template(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        tools: Any = None,
        add_generation_prompt: bool = True
    ) -> str:
        """
        トークナイザを使って、messages をチャット形式のテキストに変換
        """

        chatml_instruct_template="{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '<|im_start|>system\n' + message['content'].rstrip() + '<|im_end|>\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'<|im_start|>user\n' + message['content'].rstrip() + '<|im_end|>\n'-}}{%- else -%}{{-'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'<|im_start|>assistant\n'-}}{%- endif -%}"

        try:
            chat_prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
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



    def count_tokens(
        self,
        model_type: str,
        tokenizer: PreTrainedTokenizer,
        prompt: str = "",
        messages: List[Dict[str, str]] = []
    ) -> int:
        """
        トークン数をカウント
        prompt か messages のどちらか一方を使う
        """

        try:
            if model_type == 'mlx':
                if messages != []:
                    tokenized_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                else:
                    tokenized_input = tokenizer.tokenize(prompt)
            elif model_type == 'llama-cpp':
                if messages != []:
                    text = json.dumps(messages)                
                else:
                    text = prompt
                text = bytes(text, 'utf-8')
                tokenized_input= tokenizer.tokenize(text)

            token_length = len(tokenized_input)
            return token_length

        except Exception as e:
            raise RuntimeError(f"Failed to count tokens: {str(e)}") from e
