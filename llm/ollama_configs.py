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

    def _get_response_token_counts(self, response):
        return None

    def chat(self, messages, **kwargs):
        return super().chat(messages, **kwargs)


# Models initialization
chat_llm = FixedOllama(
    model="qwen3:14b",
    request_timeout=60.0,
    max_tokens=200,
)

code_assistant_llm = FixedOllama(
    model="deepseek-coder-v2:16b",
    request_timeout=60.0,
    max_tokens=300,
    temperature=0.5
)
