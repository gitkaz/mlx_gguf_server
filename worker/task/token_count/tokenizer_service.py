from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer
import os
import json
import re

from ...task_response import TaskResponse
from ...llm_model import LLMModel
from schemas import TokenCountParams
from ...logger_config import setup_logger

log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)


class TokenizerService:
    """
    トークナイザに関する操作を専門に行うサービス
    - チャットテンプレートの適用
    - トークン数のカウント
    """

    def get_chat_template(self, tokenizer: PreTrainedTokenizer) -> str:
        """
        Get the model's default chat template string

        Args:
            tokenizer: The tokenizer instance

        Returns:
            Chat template string

        Raises:
            ValueError: If tokenizer doesn't have a chat_template
        """
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat_template")
        return tokenizer.chat_template

    def modify_chat_template_for_thinking(
        self, 
        template: str, 
        enable_thinking: bool,
        thinking_variable: str = "enable_thinking"
    ) -> str:
        """
        Modify chat template to set thinking variable.

        Args:
            template: Original chat template string
            enable_thinking: True for reasoning mode, False for non-reasoning
            thinking_variable: Name of the template variable (default: "enable_thinking")

        Returns:
            Modified chat template string
        """
        # Check if template already has the thinking variable
        if thinking_variable in template:
            # Replace existing thinking variable setting
            pattern = r'\{%-?\s*set\s+' + re.escape(thinking_variable) + r'\s*=\s*(true|false)\s*-?%\}'
            replacement = f'{{%- set {thinking_variable} = {"true" if enable_thinking else "false"} %}}'
            modified = re.sub(pattern, replacement, template, flags=re.IGNORECASE)
            logger.debug(f"Replaced existing {thinking_variable} setting: {enable_thinking}")
            return modified
        else:
            # Add thinking variable at the beginning of the template
            thinking_directive = f'{{%- set {thinking_variable} = {"true" if enable_thinking else "false"} %}}\n'
            logger.debug(f"Added {thinking_variable} directive: {enable_thinking}")
            return thinking_directive + template


    def apply_chat_template(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        tools: Any = None,
        add_generation_prompt: bool = True,
        enable_thinking: Optional[bool] = None,
        model_capabilities: Optional[Dict] = None
    ) -> str:
        """
        トークナイザを使って、messages をチャット形式のテキストに変換
        """

        # prepare chatml_template for last resort
        chatml_instruct_template="{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '<|im_start|>system\n' + message['content'].rstrip() + '<|im_end|>\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'<|im_start|>user\n' + message['content'].rstrip() + '<|im_end|>\n'-}}{%- else -%}{{-'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'<|im_start|>assistant\n'-}}{%- endif -%}"

        # 1. Determine final template
        final_template = self.get_chat_template(tokenizer)

        # 2. Apply thinking toggle if requested and model supports it
        if (enable_thinking is not None and 
            model_capabilities and 
            model_capabilities.get('supports_thinking_toggle')):

            thinking_variable = model_capabilities.get('thinking_variable', 'enable_thinking')
            final_template = self.modify_chat_template_for_thinking(
                final_template, 
                enable_thinking,
                thinking_variable
            )
            logger.info(f"Applied thinking toggle: {thinking_variable}={enable_thinking} for model")

        # Apply the template

        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                chat_template=final_template
            )
        except Exception as e:
            logger.warning(f"Chat template failed (attempt 1): {str(e)}")
            try:
                tokenizer.chat_template = tokenizer.default_chat_template
                chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            except Exception as e2:
                logger.warning(f"Chat template failed (attempt 2): {str(e2)}")
                chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, chat_template=chatml_instruct_template)
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
