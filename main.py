import logging

from fastapi import FastAPI
from nicegui import ui

from chat.interface.main_tabs import create_main_menu
from chat.interface.chat_constructor import create_chat_page
from config.settings import (
    AppConfig,
    ClientsConfig,
    NLPConfig,
    RAGTabConfig,
    CodeAssistantTabConfig,
)
from llm.ollama_inference import ask_llm
from chat.backend.dialogue import Dialogue

LOG_PATH = "rag_chatbot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

app_config = AppConfig()
clients_config = ClientsConfig()
nlp_config = NLPConfig()
logger = logging.getLogger(__name__)

dialogue = Dialogue(
    app_config=app_config,
    clients_config=clients_config,
    nlp_config=nlp_config,
    logger=logger,
)

app: FastAPI = FastAPI()
tabs = [RAGTabConfig(), CodeAssistantTabConfig()]

create_main_menu(tabs)
for tab in tabs:
    create_chat_page(
        tab=tab,
        app=app,
        app_config=app_config,
        clients_config=clients_config,
        dialogue=dialogue,
        ask_llm=ask_llm,
        logger=logger,
    )

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=app_config.app_port, show=True)
