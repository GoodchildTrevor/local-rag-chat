import logging

from fastapi import FastAPI
from nicegui import ui

from chat.interface.chat_ui import create_chat_page
from config.settings import (
    AppConfig,
    ClientsConfig,
    EmbeddingModelsConfig,
    NLPConfig,
    RAGTabConfig,
    CodeAssistantTabConfig
)
from llm.ollama_inference import ask_llm
from chat.backend.dialogue import Dialogue

# Configure logging to both file and console
LOG_PATH = "rag_chatbot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Load application settings
app_config = AppConfig()
clients_config = ClientsConfig()
embedding_models_config = EmbeddingModelsConfig()
nlp_config= NLPConfig()
# Initialize logger
logger = logging.getLogger(__name__)
# Create the semantic search object with a vector collection
dialogue = Dialogue(
    app_config=app_config,
    clients_config=clients_config,
    embedding_models_config = embedding_models_config,
    nlp_config=nlp_config,
    logger=logger,
)
# Initialize FastAPI backend
app: FastAPI = FastAPI()

# Set up NiceGUI UI routes and layout
tabs = [RAGTabConfig(), CodeAssistantTabConfig(),]

for tab in tabs:
    create_chat_page(
        tab=tab,
        app=app,
        app_config=app_config,
        clients_config=clients_config,
        embedding_models_config=embedding_models_config,
        dialogue=dialogue,
        ask_llm=ask_llm,
        logger=logger
    )

# Run the application with NiceGUI if started as a main script
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=app_config.app_port, show=True)
