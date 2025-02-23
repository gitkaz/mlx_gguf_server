from dataclasses import dataclass
from typing import List

@dataclass
class OpenAICompatibleEmbedding:
    """
    Response Body of Embedding API. Referred by OpenAI API.
    Ref: https://platform.openai.com/docs/api-reference/embeddings/object
    """
    object: str
    embedding: List[float]
    index: int

@dataclass
class OpenAICompatibleEmbeddings:
    """
    List of OpenAICompatibleEmbedding.
    """
    embeddings: List[OpenAICompatibleEmbedding]

    def __iter__(self):
        return iter(self.embeddings)

    def __len__(self):
        return len(self.embeddings)