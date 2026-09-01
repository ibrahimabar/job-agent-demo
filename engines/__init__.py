"""Pick an engine. Everything else in the app is engine-agnostic.

    from engines import get_engine
    engine = get_engine()          # honours the ENGINE env var
    engine = get_engine("local")   # or ask for one by name

Choosing when nothing is specified: **the keyword engine**. It costs nothing,
needs no key, no network and no GPU, and it always works. Nothing here should
start spending money because you opened the app — a paid call is a decision, and
decisions are made by picking an engine on purpose.

Set ENGINE, or use the sidebar, to pick a different one:

    ENGINE=claude     streamlit run app.py     # costs money per run
    ENGINE=offline    streamlit run app.py     # recorded Claude, free
    ENGINE=local      streamlit run app.py     # Ollama, free but slow
"""

from __future__ import annotations

import os

import config

from .base import CVSource, Engine
from .claude import ClaudeEngine, OfflineEngine
from .heuristic import HeuristicEngine
from .local import LocalEngine

ENGINES: dict[str, type[Engine]] = {
    "claude": ClaudeEngine,
    "offline": OfflineEngine,
    "local": LocalEngine,
    "heuristic": HeuristicEngine,
}

# Shown in this order in the UI. The free default comes first: it is what the
# app opens on, and starting from the weakest engine makes the comparison with
# the others the natural next click.
ORDER = ["heuristic", "claude", "offline", "local"]


def get_engine(name: str | None = None) -> Engine:
    name = (name or os.environ.get("ENGINE") or "").strip().lower()
    if name in ENGINES:
        return ENGINES[name]()
    return ENGINES[default_engine_name()]()


def default_engine_name() -> str:
    """What to use when nobody said: the one that is free and always works.

    Deliberately NOT "use Claude if a key happens to be present". A key in .env
    is there so that choosing Claude works, not so that every run bills you for
    it. The free engine is also the better teaching default — students meet the
    keyword baseline first and then see what a model adds, which is the point of
    having four engines at all.
    """
    requested = os.environ.get("ENGINE", "").strip().lower()
    if requested in ENGINES:
        return requested
    return "heuristic"


def all_engines() -> list[Engine]:
    return [ENGINES[n]() for n in ORDER]


__all__ = ["CVSource", "Engine", "get_engine", "all_engines", "default_engine_name", "ENGINES"]
