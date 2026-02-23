from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class LLMModel:
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    model_type: str = ""
    model_path: str = ""
    model_name: str = ""
    default_gen_params: Dict[str, Any] = field(default_factory=dict)
    context_length: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)