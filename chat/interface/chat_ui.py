from nicegui import ui
from fastapi import FastAPI
import asyncio

from config.chat_messages import (
    ENTER,
    GREETINGS,
    MODEL_RESPOND,
    DOC_STANDBY,
    MODEL_STANDBY,
    DOC_BUG,
    MODEL_BUG,
)
from chat.interface.chat_utils import (
    search_display
)


def setup_ui(app: FastAPI, search, ask_llm, logger):
    """Setup NiceGUI interface for the RAG chatbot"""

    @ui.page("/")
    def chat_page() -> None:
        """Main page for chat"""
        # initialize empty history
        history: list[tuple[str, str]] = []
        # setup styles
        with ui.column().classes("flex-1 h-screen max-w-4xl mx-auto p-4"):
            # Header
            ui.html("<h1 class='text-3xl font-bold mb-6 text-center text-blue-600'>RAG Chatbot</h1>")
            # Chat history container
            history_ui = ui.column().classes("flex-1 overflow-y-auto p-4 bg-gray-50 rounded-lg min-h-96 mb-4")

            def render_message(text: str, *, sender: str = "", system: bool = False):
                """Renders a message in the chat history UI with avatars."""
                with history_ui:
                    if system:
                        with ui.row().classes("w-full justify-center mb-2"):
                            ui.markdown(text).classes("text-gray-600 text-sm italic")
                        return None, None

                    with ui.row().classes("items-start gap-3 mb-4 w-full"):
                        if sender == "user":
                            ui.avatar("person", color="blue").classes("shrink-0")  # User's emoji
                            with ui.column().classes("flex-1"):
                                ui.markdown(f"**Вы:** {text}").classes(
                                    "bg-blue-100 p-3 rounded-lg max-w-prose"
                                )
                            return None, None
                        else:
                            ui.avatar("smart_toy", color="green").classes("shrink-0")  # Bot's emoji
                            with ui.column().classes("flex-1"):
                                docs_md = ui.markdown(DOC_STANDBY).classes(
                                    "bg-yellow-100 p-3 rounded-lg max-w-prose mb-2"
                                )
                                answer_md = ui.markdown(MODEL_STANDBY).classes(
                                    "bg-green-100 p-3 rounded-lg max-w-prose"
                                )
                            return docs_md, answer_md

            async def send() -> None:
                """Handles sending a user message and getting a bot response."""
                msg = input_box.value.strip()
                if not msg:
                    return

                input_box.value = ""
                input_box.props(add="loading")
                send_btn.props(add="loading")
                clear_btn.props(add="disable")
                await asyncio.sleep(0)
                # Render user message
                render_message(msg, sender="user")
                # Render bot response placeholders
                docs_md, answer_md = render_message("", sender="bot")

                async def background_task():
                    """Background task to handle LLM interaction and UI updates."""
                    try:
                        logger.info(f"Processing user message: {msg}")

                        with history_ui:
                            docs_md.content = DOC_STANDBY
                            answer_md.content = ""

                        # First step: relevant docs
                        logger.debug("Attempting to call search.search_display...")
                        try:
                            results = await search.get_searching_results(msg)
                            docs, display_docs = await search_display(results, logger)
                            with history_ui:
                                docs_md.content = display_docs
                            logger.debug("search.search_display call completed.")
                        except Exception as e_search:
                            logger.error(f"Search error: {str(e_search)}", exc_info=True)
                            with history_ui:
                                docs_md.content = f"**Ошибка поиска:** {str(e_search)}<br>{DOC_BUG}"
                                answer_md.style("display: none")
                            return
                        await asyncio.sleep(0)

                        # Second step: answer generation
                        with history_ui:
                            answer_md.content = MODEL_STANDBY

                        logger.debug("Attempting to call ask_llm...")
                        try:
                            answer = await ask_llm(logger, msg, docs, history, results)
                            with history_ui:
                                answer_md.content = f"{MODEL_RESPOND} {answer}"
                            history.append((msg, str(answer)))
                            logger.info(f"Successfully processed message: {str(answer)}")
                            logger.debug("ask_llm call completed.")
                        except Exception as e_model:
                            logger.error(f"Model error: {str(e_model)}", exc_info=True)
                            with history_ui:
                                answer_md.content = f"**Ошибка модели:** {str(e_model)}<br>{MODEL_BUG}"
                    finally:
                        input_box.props(remove="loading")
                        send_btn.props(remove="loading")
                        clear_btn.props(remove="disable")

                        def scroll_to_bottom():
                            """Auto scroll down after new message"""
                            ui.run_javascript("""
                                const container = document.querySelector('.overflow-y-auto');
                                if (container) {
                                    container.scrollTop = container.scrollHeight;
                                }
                            """)

                        with history_ui:
                            ui.timer(0.1, scroll_to_bottom, once=True)

                asyncio.create_task(background_task())

            async def clear_history() -> None:
                """Clears the chat history and updates the UI."""
                history.clear()
                history_ui.clear()  # delete all previous messages
                render_message(GREETINGS, system=True)  # render greeting message
                ui.notify('История чата очищена.', type='info')  # notify about cleaning
                logger.info("Chat history cleared.")
            # Input section
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full gap-2 items-center"):
                    input_box = (
                        ui.input(placeholder="Введите ваш вопрос...")
                        .props("rounded outlined")
                        .classes("flex-1")
                    )

                    send_btn = ui.button(ENTER, on_click=lambda: asyncio.create_task(send()))
                    send_btn.props("color=primary")
                    clear_btn = ui.button("Очистить историю", on_click=lambda: asyncio.create_task(clear_history()))
                    clear_btn.props("color=positive")
                    # Enter key handler
                    input_box.on("keydown.enter", lambda e: asyncio.create_task(send()))
            # Initial system message
            render_message(GREETINGS, system=True)

    @ui.page("/chat")
    def chat_redirect():
        """Redirect /chat to main page"""
        ui.navigate.to("/")

    @ui.page("/api")
    def api_info():
        """API information page"""
        ui.html("<h1>API Information</h1>")
        ui.markdown("""
        ### Available Endpoints:
        - GET / - Main chat interface
        - POST /api/chat - Chat API endpoint
        - GET /chat - Redirects to main page
        """)

    return app
