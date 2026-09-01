"""The Engine interface — one seam, four ways to fill it.

The workshop app needs five things done. Each is a place where judgment is
required, and each can be done by a frontier model, by a small local model, or
by no model at all:

    summarize_cv    read a CV and say who this person is
    extract_profile turn that CV into structured data
    run_agent       search for jobs, decide what to look at, report back
    rank            judge each role against this candidate
    draft           write a tailored application

An Engine implements all five. Swap the engine, and every other line of the app
stays the same. That is the actual lesson of this file: put the model behind an
interface and it becomes a choice rather than a dependency.

Engines available:

    claude      the real thing. Anthropic API, needs a key.
    offline     the same responses, replayed from data/cache/. Needs a recording.
    local       a model on your own machine via Ollama. Free, slower, weaker.
    heuristic   no model at all. Regex and keyword scoring. Free, instant, dumb.

Read heuristic.py and claude.py side by side. The diff between them is worth
more than any slide about what LLMs are for.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CVSource:
    """A CV, in whatever form the engine needs it.

    Claude takes the PDF bytes directly. The heuristic and local engines need
    plain text, so they get it extracted — which is exactly the parsing problem
    the Claude path lets you skip.
    """

    raw: bytes
    filename: str

    @classmethod
    def from_path(cls, path: str | Path) -> "CVSource":
        path = Path(path)
        return cls(raw=path.read_bytes(), filename=path.name)

    @property
    def is_pdf(self) -> bool:
        return self.filename.lower().endswith(".pdf")

    def as_anthropic_block(self) -> dict:
        """The content block Claude reads. No parsing involved."""
        if self.is_pdf:
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.standard_b64encode(self.raw).decode("utf-8"),
                },
            }
        return {"type": "text", "text": self.raw.decode("utf-8", errors="replace")}

    def as_text(self) -> str:
        """Plain text, for engines that cannot read a PDF themselves.

        This is the code the Claude path does not need to exist. It is also
        worse: pypdf returns columns interleaved, drops table structure, and
        makes a mess of anything designed rather than typed. Compare the output
        of this with what `summarize_cv` gets from the document block.
        """
        if not self.is_pdf:
            return self.raw.decode("utf-8", errors="replace")

        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(self.raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            return f"[could not extract text from {self.filename}: {exc}]"


def split_search_modes(search_modes) -> tuple[bool, bool, list[str]]:
    """Turn what the user ticked into (want_israel, want_remote, cities).

    The sidebar mixes regions and specific cities in one list — "Remote",
    "Israel — anywhere", "Herzliya" — because that is how people think about
    where they will work. Engines get it separated out here rather than each
    one reimplementing the parsing.
    """
    modes = list(search_modes or [])
    want_remote = any(m.lower().startswith("remote") for m in modes)
    cities = [m for m in modes
              if not m.lower().startswith("remote") and not m.lower().startswith("israel")]
    want_israel = any(m.lower().startswith("israel") for m in modes) or bool(cities)
    return want_israel, want_remote, cities


class Engine:
    """What every engine must provide."""

    name: str = "base"
    label: str = "Base"
    blurb: str = ""
    costs_money: bool = False

    # --- availability --------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        """(usable right now?, why not / what it needs)."""
        return True, ""

    # --- the five jobs -------------------------------------------------------

    def summarize_cv(self, cv: CVSource) -> str:
        raise NotImplementedError

    def extract_profile(self, cv: CVSource):
        raise NotImplementedError

    def run_agent(self, profile, *, on_event=None, search_modes=("Israel", "Remote")) -> dict:
        raise NotImplementedError

    def rank(self, profile, jobs: list[dict], *, top_n: int = 8, on_progress=None):
        raise NotImplementedError

    def draft(self, profile, job: dict, match=None) -> dict:
        raise NotImplementedError
