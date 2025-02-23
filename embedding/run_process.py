from typing import List
from .embedding_schemas import OpenAICompatibleEmbedding, OpenAICompatibleEmbeddings
from transformers import AutoModel, AutoTokenizer
import numpy as np

class TokenLimitExceededError(Exception):
    """
    Exception for over jina-embeddings-v3's token length (8192)
    """
    def __init__(self, index: int = 0, length: int = 0):
        super().__init__(f"Error at embedding. List[{index}] token length {length} over 8192.")
        self.index = index
        self.length = length

def run(params: dict, queue):
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

    try:
        texts = params["input"]
        dimensions = params["dimensions"]
        if isinstance(texts, str):
            texts = [texts]

        openai_embeddings: List[OpenAICompatibleEmbedding] = []
 
        # check token lengh
        for i, text in enumerate(texts): 
            tokens = tokenizer.tokenize(text)
            token_length = len(tokens)
            if token_length > 8192:
                queue.put(TokenLimitExceededError(index=i, length=token_length))
                return

        # embedding
        if dimensions is None:
            embeddings = model.encode(texts, task="text-matching")
        else:
            embeddings = model.encode(texts, task="text-matching", truncate_dim=dimensions)

        for embedding in embeddings:
            # embedding は NumPy 配列なので、float のリストに変換
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            openai_embedding = OpenAICompatibleEmbedding(
                object="embedding", embedding=embedding, index=i
            )
            openai_embeddings.append(openai_embedding)

        result = OpenAICompatibleEmbeddings(embeddings=openai_embeddings)
        queue.put(result)

    except Exception as e:
        print(f"Error at embedding: {e}")
        queue.put(e)