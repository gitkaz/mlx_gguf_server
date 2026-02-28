from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer
import os

from schemas import InputTokenCountParams
from ...task_response import TaskResponse
from ...llm_model import LLMModel
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


    def count_tokens(self, llm_model: LLMModel, params) -> TaskResponse:
        """
        Count tokens for input.
        """
        
        if not isinstance(llm_model, LLMModel):
            raise TypeError("First argument must be LLMModel instance")

        # Handle new OpenAI-compatible InputTokenCountParams
        if isinstance(params, InputTokenCountParams):
            if llm_model.model_type != 'mlx':
                raise RuntimeError(f"Only MLX models are supported. Got: {llm_model.model_type}")

            if not llm_model.tokenizer:
                raise RuntimeError("Tokenizer is not loaded")

            try:
                # Simple tokenization - no chat template conversion
                tokenized_input = llm_model.tokenizer.encode(params.input)
                token_length = len(tokenized_input)
                return TaskResponse(200, token_length)
            except Exception as e:
                raise RuntimeError(f"Failed to count tokens: {str(e)}") from e
