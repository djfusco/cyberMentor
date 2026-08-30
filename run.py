"""Launches the Practice-only beta build of the AI Exercise Mentor.

This file is exported to the beta repo AS `run.py` (see
scripts/beta_manifest.txt), so beta testers just run: python run.py

This is a new, additive file -- the normal run.py (which points at the full
app/main.py) is unchanged and unaffected.
"""
import logging

import uvicorn

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    settings = get_settings()
    print()
    print("AI Exercise Mentor (beta)")
    print()
    print(f"http://{settings.host}:{settings.port}")
    print()
    print("This beta build only includes the Practice flow: view/start a lab,")
    print("chat with the mentor during an exercise, and see your results.")
    print()
    print(f"Ollama must be running separately at {settings.ollama_base_url} (start with: ollama serve)")
    print()
    uvicorn.run("app.main_beta:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
