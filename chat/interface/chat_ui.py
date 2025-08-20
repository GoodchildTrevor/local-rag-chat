import asyncio
from logging import Logger
from typing import Callable
import uuid

from nicegui import ui
from fastapi import FastAPI

from config.consts.chat_messages import (
    ENTER, GREETINGS, MODEL_RESPOND, 
    DOC_STANDBY, MODEL_STANDBY,
    DOC_BUG, MODEL_BUG, THANKS, STARS
)
from config.consts.tab_config import TabConfig
from config.settings import (
    AppConfig,
    ClientsConfig,
    EmbeddingModelsConfig,
)
from chat.interface.chat_utils import (
    answer_display, 
    search_display, 
)
from chat.backend.dialogue import Dialogue  
from databases.cashing.cashing import AnswerCash


def create_chat_page(
        tab: TabConfig,
        app: FastAPI,
        app_config: AppConfig,
        clients_config: ClientsConfig,
        embedding_models_config: EmbeddingModelsConfig,
        dialogue: Dialogue,
        ask_llm: Callable,
        logger: Logger
):
    """Setup NiceGUI interface for the RAG chatbot"""
    prefix = tab.prefix
    header = tab.header
    system_prompt = tab.system_prompt
    llm = tab.llm

    sessions: dict[str, AnswerCash] = {}

    @ui.page(f"/{prefix}")
    def chat_page() -> None:
        """Main page for chat"""
        history: list[tuple[str, str]] = []
        session_id = f"{prefix}_{str(uuid.uuid4())}"
        answer_cash = AnswerCash(
            logger=logger,
            clients_config = clients_config,
            embedding_model_config=embedding_models_config,
            collection_name=app_config.cash_collection,
            timeout_minutes=app_config.timeout,
            session_id=session_id,
        )
        sessions[session_id] = answer_cash

        async def auto_flush():
            await asyncio.sleep(app_config.timeout * 60)
            logger.info(f"Auto-flushing session: {session_id}")
            await answer_cash.flush()

        asyncio.create_task(auto_flush())

        with ui.column().classes("flex-1 h-screen w-[80%] mx-auto p-4"):
            logger.info(header)
            ui.label(header).classes("text-3xl font-bold mb-6 text-center text-blue-600")
            history_ui = ui.column().classes("flex-1 flex-grow w-[100%] overflow-y-auto p-4 bg-gray-50 rounded-lg mb-4")

            def render_message(text: str, *, sender: str = "", system: bool = False) -> tuple:
                """Render message within explicit history_ui context"""
                with history_ui:
                    if system:
                        with ui.row().classes("w-full justify-center mb-2"):
                            ui.markdown(text).classes("text-gray-600 text-sm italic")
                        return None, None
                    
                    if sender == "user":
                        with ui.row().classes("items-start gap-3 mb-4 w-full"):
                            ui.avatar("person", color="blue").classes("shrink-0")
                            with ui.column().classes("flex-1"):
                                ui.markdown(f"**Вы:** {text}").classes("bg-blue-100 p-3 rounded-lg max-w-prose")
                        return None, None
                    else:
                        with ui.row().classes("items-start gap-3 mb-4 w-full"):
                            ui.avatar("smart_toy", color="green").classes("shrink-0")
                            with ui.column().classes("flex-1"):
                                docs_md = ui.markdown(DOC_STANDBY).classes("bg-yellow-100 p-3 rounded-lg max-w-prose mb-2")
                                answer_md = ui.markdown(MODEL_STANDBY).classes("bg-green-100 p-3 rounded-lg max-w-prose")
                        return docs_md, answer_md

            async def send() -> None:
                msg = input_box.value.strip()
                if not msg:
                    return

                input_box.value = ""
                input_box.props(add="loading")
                send_btn.props(add="loading")
                clear_btn.props(add="disable")
                await asyncio.sleep(0)

                # Render messages within explicit history_ui context
                with history_ui:
                    render_message(msg, sender="user")
                    docs_md, answer_md = render_message("", sender="bot")

                asyncio.create_task(background_task(msg, docs_md, answer_md))
            
            async def background_task(msg: str, docs_md: ui.markdown, answer_md: ui.markdown):
                """Performs RAG search and LLM query in the background."""
                # The 'with' statement here correctly enters the UI container's slot for this task
                with history_ui:
                    try:
                        logger.info(f"Processing user message: {msg}")
                        # Update initial placeholder messages
                        if prefix == "chat":
                            docs_md.content = DOC_STANDBY
                            answer_md.content = ""
                            priority_results=False
                            docs = list()
                            results = list()

                            try:
                                # Perform search and check cache concurrently
                                normalized_msg = dialogue.processing_query(msg)
                                search_results_task = dialogue.get_searching_results(
                                    collection=app_config.rag_collection, 
                                    normalized_query=normalized_msg
                                )
                                best_answers_task = dialogue.get_cashed_answers(
                                    embedding_models_config=embedding_models_config,
                                    collection=app_config.cash_collection, 
                                    normalized_query=normalized_msg
                                )
                                results, priority_results = await asyncio.gather(search_results_task, best_answers_task)

                                if priority_results:
                                    logger.debug("Using cached answers")
                                    results_to_use = priority_results
                                    docs, display_docs = await answer_display(results_to_use)
                                    question_id = priority_results[0].payload.get("question_id")
                                else:
                                    logger.debug("Using new search results")
                                    results_to_use = results
                                    docs, display_docs = await search_display(results_to_use, logger)
                                    question_id = None
                                
                                # Update the UI with search results
                                docs_md.content = display_docs

                            except Exception as e_search:
                                logger.error(f"Search error: {str(e_search)}", exc_info=True)
                                docs_md.content = f"**Ошибка поиска:** {str(e_search)}<br>{DOC_BUG}"
                                answer_md.style("display: none")
                                return

                        # Update LLM placeholder
                        answer_md.content = MODEL_STANDBY

                        try:
                            if priority_results:
                                answer = priority_results[0].payload.get("document")
                            else: 
                                answer = await ask_llm(
                                    logger=logger,
                                    llm=llm,
                                    system_prompt=system_prompt, 
                                    query=msg, 
                                    context=docs,
                                    history=history, 
                                    results=results,
                                )
                            
                            # Update the UI with the final answer
                            answer_md.content = f"{MODEL_RESPOND} {answer}"

                            def on_rating_change(rate):
                                rating_value = rate.value
                                logger.info(f"User gave {rating_value} stars")
                                ui.notify(THANKS)
                                
                                # Save the rating and answer to the cache
                                async def cache_rating():
                                        
                                        await answer_cash.add(
                                            user_id=0,
                                            question_id=question_id,
                                            msg=normalized_msg,
                                            answer=answer,
                                            display_docs=display_docs,
                                            rating=rating_value,
                                        )

                                asyncio.create_task(cache_rating())
                            
                            ui.rating(
                                value=0.0, max=STARS, color="yellow", size="lg",
                                on_change=on_rating_change
                            )

                            history.append((msg, str(answer)))
                            logger.info(f"Successfully processed message: {str(answer)}")

                        except Exception as e_model:
                            logger.error(f"Model error: {str(e_model)}", exc_info=True)
                            answer_md.content = f"**Ошибка модели:** {str(e_model)}<br>{MODEL_BUG}"
                    
                    finally:
                        # Ensure loading states are removed and UI scrolls to the bottom
                        input_box.props(remove="loading")
                        send_btn.props(remove="loading")
                        clear_btn.props(remove="disable")

                        def scroll_to_bottom():
                            ui.run_javascript("""
                                const container = document.querySelector('.overflow-y-auto');
                                if (container) {
                                    container.scrollTop = container.scrollHeight;
                                }
                            """)
                        
                        ui.timer(0.1, scroll_to_bottom, once=True)

            async def clear_history() -> None:
                """Clear chat history"""
                history.clear()
                history_ui.clear()
                
                with history_ui:
                    render_message(GREETINGS, system=True)
                    ui.notify('История чата очищена.', type='info')
                
                logger.info("Chat history cleared.")
                await answer_cash.flush(immidiate= True)

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
                    input_box.on("keydown.enter", lambda e: asyncio.create_task(send()))

            render_message(GREETINGS, system=True)


    @ui.page("/")
    def main_menu():
        ui.label("Главное меню").classes("text-2xl font-bold")
        with ui.column().classes("gap-4 mt-4"):
            ui.button("Чат-бот", on_click=lambda: ui.navigate.to("/chat")).classes("w-48")
            ui.button("Код-ассистент", on_click=lambda: ui.navigate.to("/assistant")).classes("w-48")


    @ui.page("/api")
    def api_info():
        ui.html("<h1>API Information</h1>")
        ui.markdown("""
        ### Available Endpoints:
        - GET / - Main menu interface
        - POST /api/chat - Chat API endpoint
        - GET /chat - QA chat interface
        """)

    return app
