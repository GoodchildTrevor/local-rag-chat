import ollama
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
import numpy as np
from typing import Any, Iterable, Union

PROMPT_TEMPLATE = PromptTemplate("""
    Системное сообщение:
    {system}

    Контекст (результаты поиска):
    {context}

    История диалога:
    {history}

    Вопрос:
    {query}
""".strip())


class OllamaDenseEmbedding:
    """
    Custom methods override for fastembed methods
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(
        self,
        documents: str | Iterable[str],
    ) -> Iterable[np.ndarray]:
        """
        Method override for common API using
        :param documents: A single document (str) or an iterable of documents (each a str) to embed.
        :returns: An iterable of NumPy arrays: each element is the embedding vector for the
        corresponding input document. Vectors are ``np.float32``.
        """
        if isinstance(documents, str):
            documents = [documents]

        embeddings = []
        for doc in documents:
            resp = ollama.embeddings(model=self.model_name, prompt=doc)
            embeddings.append(np.array(resp["embedding"], dtype=np.float32))

        return embeddings
    
    def query_embed(
        self, 
        query: Union[str, Iterable[str]], 
        **kwargs: Any
    ) -> Iterable[np.ndarray]:
        """
        Embeds queries, method override for common API
        query: The query to embed, or an iterable e.g. list of queries
        :returns: An iterable of NumPy arrays: each element is the embedding vector for the
        corresponding input document. Vectors are ``np.float32``.
        """
        if isinstance(query, str):
            yield from self.embed([query], **kwargs)
        else:
            yield from self.embed(query, **kwargs)


# Models initialization
chat_llm = Ollama(
    model="qwen3:14b",
    request_timeout=60.0,
    max_tokens=200,
)

code_assistant_llm = Ollama(
    model="deepseek-coder-v2:16b",
    request_timeout=60.0,
    max_tokens=300,
    temperature=0.5
)
