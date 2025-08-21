import functools

from nicegui import ui

from config.consts.chat_messages import MAIN_MENU
from config.consts.tab_config import TabConfig


def create_main_menu(tabs: list[TabConfig]):
    """Create main menu page"""
    @ui.page("/")
    def main_menu():

        with ui.column().classes("h-screen w-full justify-center items-center gap-8"):

            ui.label(MAIN_MENU).classes("text-6xl font-bold mb-10")
                
            with ui.column().classes("w-[25%] h-[15%] items-center gap-4"):
                for tab in tabs:
                    on_click = functools.partial(ui.navigate.to, f"/{tab.prefix}")
                    with ui.button(f"{tab.header}", on_click=on_click) \
                        .classes("w-[100%] h-[100%] justify-start text-left") \
                        .props("color=primary"):
                        ui.icon(f"{tab.prefix}").classes("mr-3")


def create_api_info():
    """Create API info page"""
    @ui.page("/api")
    def api_info():
        with ui.column().classes("h-screen w-full justify-center items-center p-8"):
            ui.html("<h1 class='text-3xl font-bold mb-6'>API Information</h1>")
            ui.markdown("""
            ### Available Endpoints:
            - GET / - Main menu interface
            - POST /api/chat - Chat API endpoint  
            - GET /chat - QA chat interface
            """).classes("text-left")
