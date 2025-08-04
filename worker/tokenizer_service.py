from typing import Dict, Any, List
from transformers import PreTrainedTokenizer
import os
import json

from .task_response import TaskResponse
from .llm_model import LLMModel
from schemas import TokenCountParams


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
        except Exception as e:
            logger.warning(f"Chat template failed (attempt 1): {str(e)}")
            try:
                tokenizer.chat_template = tokenizer.default_chat_template
                chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                logger.debug(f"{chat_prompt=}")
            except Exception as e2:
                logger.warning(f"Chat template failed (attempt 2): {str(e2)}")
                chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chatml_instruct_template)
                logger.debug(f"{chat_prompt=}")
        return chat_prompt

    def count_tokens(self, llm_model: LLMModel, params: TokenCountParams) -> TaskResponse:
        """
        トークン数をカウント
        prompt か messages のどちらか一方を使う
        """
        if not isinstance(llm_model, LLMModel):
            raise TypeError("First argument must be LLMModel instance")

        model_type = llm_model.model_type
        tokenizer = llm_model.tokenizer
        prompt = params.prompt
        messages = params.messages

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
            return TaskResponse(200, token_length)

        except Exception as e:
            raise RuntimeError(f"Failed to count tokens: {str(e)}") from e
