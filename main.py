import logging
import os

from dotenv import load_dotenv

from fastapi import FastAPI
from nicegui import ui

from chat.interface.chat_ui import setup_ui
from config.settings import (
    get_settings,
    ChatRequest
)
from models.ollama_inference import ask_llm
from chat.backend.dialogue import Search

load_dotenv()

collection = os.getenv("RAG_DOC_COLLECTION")

# Configure logging to both file and console
LOG_PATH = "backend/rag_chatbot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Load application settings
settings = get_settings()
# Initialize logger
logger: logging.Logger = logging.getLogger(__name__)
# Create the semantic search object with a vector collection
search: Search = Search(
    logger=logger,
    collection=collection,
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
    app,
    search,
    ask_llm,
    logger
)

# Run the application with NiceGUI if started as a main script
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=settings.app_port, show=True)
