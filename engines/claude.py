"""The real engine: Anthropic's API, and the same code the workshop teaches.

This file is thin on purpose. All it does is adapt the existing modules —
profile.py, agent.py, scoring.py, apply.py — to the Engine interface. Nothing
about how the workshop works changed to accommodate the other engines, which is
the point: the abstraction was added around the real implementation, not
instead of it.

Two engines are registered from here:

    claude    live API calls. Costs money, best quality.
    offline   the identical code path, replaying data/cache/. Free, but only
              covers requests that were recorded — change a prompt or a CV and
              you fall off the recording.
"""

from __future__ import annotations

import apply as apply_module
import config
import llm
import profile as profile_module
import scoring
import tools as tool_module
from agent import SYSTEM, run_agent

from .base import CVSource, Engine


class ClaudeEngine(Engine):
    name = "claude"
    label = "Claude (Anthropic API)"
    blurb = "The real thing. Needs an API key and costs a few cents per run."
    costs_money = True
    offline = False

    def _set_mode(self) -> None:
        """llm.py reads OFFLINE at call time, so the engine can flip it per run."""
        llm.OFFLINE = self.offline

    def available(self) -> tuple[bool, str]:
        if not config.have_api_key():
            return False, ("No ANTHROPIC_API_KEY. Copy .env.example to .env and paste a key, "
                           "or pick a free engine.")
        return True, ""

    def summarize_cv(self, cv: CVSource) -> str:
        self._set_mode()
        return profile_module.summarize_cv(cv.as_anthropic_block())

    def extract_profile(self, cv: CVSource):
        self._set_mode()
        return profile_module.extract_profile(cv.as_anthropic_block())

    def run_agent(self, profile, *, on_event=None, search_modes=("Israel", "Remote")) -> dict:
        self._set_mode()
        tool_module.set_profile(profile)
        where = ", ".join(search_modes) if search_modes else "wherever they can work"
        instruction = (
            f"{profile_module.profile_brief(profile)}\n\n"
            f"Find this person roles worth their time. Look at: {where}. "
            f"Where a specific city is named, pass it as the `location` argument to "
            f"search_israel_jobs rather than filtering afterwards. "
            f"Search more than once, open the promising postings, and report back."
        )
        return run_agent(instruction, system=SYSTEM,
                         tools=tool_module.TOOLS_WITH_APPLY, on_event=on_event)

    def rank(self, profile, jobs: list[dict], *, top_n: int = 8, on_progress=None):
        self._set_mode()
        return scoring.rank_jobs(profile, jobs, top_n=top_n, on_progress=on_progress)

    def draft(self, profile, job: dict, match=None) -> dict:
        self._set_mode()
        return apply_module.draft_application(profile, job, match)


class OfflineEngine(ClaudeEngine):
    name = "offline"
    label = "Claude, replayed (offline)"
    blurb = "Recorded Claude responses from data/cache/. Free, but only the scripted path."
    costs_money = False
    offline = True

    def available(self) -> tuple[bool, str]:
        count = len(list(config.CACHE_DIR.glob("*.json"))) if config.CACHE_DIR.exists() else 0
        if not count:
            return False, ("data/cache/ is empty. Someone has to record it once with a key:\n"
                           "    RECORD=1 python record_cache.py")
        return True, ""
