"""Mentor-chat-backend status formatting, mirroring
app/services/evidence_status.py.

MENTOR_CHAT_PROVIDER (see app/config.py) lets mentor chat use either local
Ollama (default) or an opt-in frontier API key (app/services/frontier_chat.py).
Every place that reports "is the chat backend up" (the browser status bar
in app/main.py and app/main_beta.py, and the terminal startup banners in
run.py and run_beta.py) must reflect whichever backend is actually
configured -- not hardcode an assumption that it's always Ollama. This
module is the single place that does that, given an already-constructed
chat backend (see app/dependencies.py::get_mentor_chat_backend).
"""
from typing import Any, Dict

from app.services.frontier_chat import FrontierChatService


async def get_chat_status(chat_backend) -> Dict[str, Any]:
    connected = await chat_backend.health()

    if isinstance(chat_backend, FrontierChatService):
        hint = None
        if not connected:
            hint = (
                "MENTOR_CHAT_PROVIDER is set to 'frontier' but no valid "
                "FRONTIER_API_KEY/FRONTIER_PROVIDER is configured. Set "
                "FRONTIER_API_KEY in .env, or set MENTOR_CHAT_PROVIDER=ollama "
                "to use local Ollama instead."
            )
        return {
            "label": f"Frontier ({chat_backend.provider})",
            "connected": connected,
            "url": chat_backend.base_url,
            "model": chat_backend.model,
            "hint": hint,
        }

    # Default: OllamaService (or anything else exposing the same interface).
    return {
        "label": "Ollama",
        "connected": connected,
        "url": chat_backend.base_url,
        "model": chat_backend.model,
        "hint": None if connected else "Start it with: ollama serve",
    }
