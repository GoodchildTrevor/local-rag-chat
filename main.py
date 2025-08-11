import logging

from fastapi import FastAPI
from nicegui import ui

from chat.interface.chat_ui import setup_ui
from config.settings import (
    AppConfig,
    ChatRequest,
    ClientsConfig,
    EmbeddingModelsConfig,
    NLPConfig,
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


@app.get("/")
def root():
    return {"message": "RAG Chatbot is running with NiceGUI frontend."}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    API endpoint to handle chat requests from the frontend.
    :param req: question and history payload
    :return: generated model response
    """
    return await ask_llm(req.question, req.history)

# Set up NiceGUI UI routes and layout
setup_ui(
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
