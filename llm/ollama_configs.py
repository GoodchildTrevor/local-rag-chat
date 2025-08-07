from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

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


class FixedOllama(Ollama):
    def __init__(self, *args, **kwargs):
        kwargs["additional_kwargs"] = kwargs.get("additional_kwargs", {})
        kwargs["additional_kwargs"]["use_gpu"] = True
        super().__init__(*args, **kwargs)

    def _get_response_token_counts(self, response):
        return None

    def chat(self, messages, **kwargs):
        return super().chat(messages, **kwargs)


# Model initialization
llm = FixedOllama(
    model="qwen3:4b",
    request_timeout=60.0,
    max_tokens=240,
)
