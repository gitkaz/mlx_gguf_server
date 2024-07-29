from pydantic import BaseModel, model_validator, Field
from typing_extensions import Self

class CompletionParams(BaseModel):
    model: str = "dummy"
    prompt: str = ""
    messages: list[dict] = []
    max_tokens: int = 4096
    temperature: float = 0.8
    seed: int = None
    stream: bool = False
    apply_chat_template: bool = False
    complete_text: bool = False
    top_p:float = 1.0
    stop:list = []
    repetition_penalty: float = None    # mlx only
    repetition_context_size: int = 20   # mlx only
    top_k: int = 0                      # llama-cpp only
    min_p: float = 0.05                 # llama-cpp only
    typical_p: float = 1.0              # llama-cpp only
    stop: list = []                     # llama-cpp only
    frequency_penalty: float = 0.0      # llama-cpp only
    presence_penalty: float = 0.0       # llama-cpp only
    repet_penalty: float = 1.1          # llama-cpp only
    top_k: int = 40                     # llama-cpp only
    mirostat_mode: int = 0              # llama-cpp only
    mirostat_tau: float = 5.0           # llama-cpp only
    mirostat_eta: float = 0.1           # llama-cpp only

    @model_validator(mode='after')
    def validate_prompt_and_messages(self) -> Self:
        prompt = self.prompt
        messages = self.messages
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        return self

class TokenCountParams(BaseModel):
    model: str = "dummy"
    prompt: str = ""
    messages: list[dict] = []

    @model_validator(mode='after')
    def validate_prompt_and_messages(self) -> Self:
        prompt = self.prompt
        messages = self.messages
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        return self

class CacheLimitParams(BaseModel):
    model: str = "dummy"
    cache_limit: int = 0


class ModelLoadParams(BaseModel):
    llm_model_name: str
    llm_model_path: str = Field(default="", exclude=True)
    chat_format: str = None # llama-cpp only

class ProcessCleanParams(BaseModel):
    timeout: int
