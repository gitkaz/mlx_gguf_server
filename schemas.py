from pydantic import BaseModel, model_validator, Field
from typing_extensions import Self
from typing import Optional, Dict, List, Union, Literal

class CompletionParams(BaseModel):
    model: str = "dummy"
    prompt: str = ""
    messages: List[Dict] = []
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stream: bool = False
    apply_chat_template: bool = False
    complete_text: bool = False
    top_p: Optional[float] = None
    stop: Optional[list] = []
    logit_bias: Optional[Dict[int, float]] = None # mlx only
    repetition_penalty: Optional[float] = None    # mlx only
    repetition_context_size: Optional[int] = 20   # mlx only
    use_kv_cache: bool = False          # mlx only
    tools: Optional[list] = None        # mlx only
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
    cache_limit: int = 0

class ModelLoadParams(BaseModel):
    llm_model_name: str
    llm_model_path: str = Field(default="", exclude=True)
    chat_format: str = None # llama-cpp only
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None
    top_p: Optional[float] = None

class ProcessCleanParams(BaseModel):
    timeout: int

class KokoroTtsParams(BaseModel):
    text: str
    lang_code: str    = "a"
    voice: str        = "af_heart"
    speed: int        = 1
    split_pattern:str = r'\n+'


class EmbeddingsParams(BaseModel):
    """
    Parameters of Embedding API. Referred by OpenAI API.
    Refs:
        https://platform.openai.com/docs/api-reference/embeddings/create
    """
    input: Union[str, List[str]] = Field(
        ...,
        description="Input text to embed, encoded as a string or array of strings. "
                    "To embed multiple inputs in a single request, pass an array of strings. "
                    "The input must not exceed the max input tokens for the model."
    )
    encoding_format: Optional[str] = Field(
        default="float",
        description="The format to return the embeddings in. Can be either float or base64."
    )
    dimensions: Optional[Literal[32, 64, 128, 256, 512, 768, 1024]] = Field(
        default=None,
        description="The number of dimensions the resulting output embeddings should have."
    )
