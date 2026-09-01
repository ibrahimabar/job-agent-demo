"""Turn a CV into a CandidateProfile — the workshop's Step 1 and Step 2.

Two ideas live here:

1. You do not need a PDF parsing library. Send the PDF to Claude as a
   `document` content block and it reads it. No pypdf, no pdfminer, no
   layout-mangled text, no "why did the two-column CV come out interleaved".

2. Prose is not data. `messages.parse()` with a Pydantic model turns the CV
   into a validated Python object you can branch on, sort by, and pass to a
   search function. That conversion is what makes the rest of the app possible.
"""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

import llm

# --- The schema --------------------------------------------------------------


class CandidateProfile(BaseModel):
    """What we need to know about a candidate in order to search for them.

    Every field description is a prompt. The model reads these, so write them
    as instructions to a careful assistant, not as notes to yourself.
    """

    name: str = Field(description="Candidate's full name, or 'Unknown' if not stated.")
    headline: str = Field(
        description="One line describing who this person is professionally, "
        "e.g. 'Senior backend engineer, 8 years, Python/Go, fintech'."
    )
    years_experience: float = Field(
        description="Total years of professional experience. Estimate from the "
        "employment dates if not stated outright. Use 0 for students/new grads."
    )
    seniority: str = Field(
        description="One of: intern, junior, mid, senior, staff, principal, manager, director."
    )
    core_skills: list[str] = Field(
        description="Technologies, languages and tools this person can actually "
        "do the job with. 8-15 items, most important first. Skip buzzwords they "
        "only listed once with no supporting experience."
    )
    domains: list[str] = Field(
        description="Industries or problem domains they have worked in, "
        "e.g. 'fintech', 'cybersecurity', 'adtech', 'healthcare'."
    )
    target_roles: list[str] = Field(
        description="3-5 job titles this person should realistically be searching "
        "for right now. Use titles as they actually appear on job boards."
    )
    search_keywords: str = Field(
        description="A short space-separated list of 4-8 lowercase terms to feed "
        "a keyword job search. Concrete technologies and role words only — no "
        "punctuation, no soft skills."
    )
    languages: list[str] = Field(description="Spoken/written languages, e.g. ['English', 'Hebrew'].")
    open_to_remote: bool = Field(description="True unless the CV says otherwise.")
    locations: list[str] = Field(
        description="Places this candidate can work from, inferred from their "
        "address and work history, e.g. ['Tel Aviv', 'Israel', 'Remote']."
    )
    notable: list[str] = Field(
        description="2-4 things a recruiter would find genuinely impressive. "
        "Specific achievements, not adjectives."
    )


# --- Loading a CV ------------------------------------------------------------


def cv_content_block(path: str | Path) -> dict:
    """Build the content block for a CV file.

    PDFs go over as a `document` block, base64-encoded. Plain text and markdown
    go over as text. Note the base64 string must have no newlines — b64encode
    already gives us that, but a hand-rolled encoder often will not.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
        }

    if suffix in (".txt", ".md", ""):
        return {"type": "text", "text": path.read_text(encoding="utf-8")}

    raise ValueError(f"Unsupported CV format '{suffix}'. Use PDF, TXT or MD.")


def cv_content_block_from_bytes(raw: bytes, filename: str) -> dict:
    """Same thing, for a Streamlit upload that never touches disk."""
    if filename.lower().endswith(".pdf"):
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(raw).decode("utf-8"),
            },
        }
    return {"type": "text", "text": raw.decode("utf-8", errors="replace")}


# --- Step 1: just read it ----------------------------------------------------

SUMMARY_SYSTEM = """You are a technical recruiter reading a CV for the first time.

Give a short, honest read: who this person is, what they are clearly strong at, \
and what is missing or unclear. Be specific and concrete. Do not flatter, and do \
not pad — four or five sentences is plenty."""


def summarize_cv(cv_block: dict) -> str:
    """Step 1: the smallest useful thing — one API call, plain text back."""
    response = llm.create(
        label="step1:cv summary",
        system=SUMMARY_SYSTEM,
        messages=[
            {
                "role": "user",
                # The document block goes BEFORE the text instruction. The model
                # reads in order, and it works better having seen the CV first.
                "content": [cv_block, {"type": "text", "text": "Read this CV."}],
            }
        ],
    )
    return llm.text_of(response)


# --- Step 2: turn it into data ----------------------------------------------

PROFILE_SYSTEM = """You extract structured candidate profiles from CVs.

Base every field on evidence in the document. When something is not stated, \
infer the most reasonable value from what is there rather than leaving it empty \
— an empty field breaks the search that runs next. Never invent employers, \
degrees, or dates."""


def extract_profile(cv_block: dict) -> CandidateProfile:
    """Step 2: same input, but now we get an object instead of a paragraph."""
    return llm.parse(
        CandidateProfile,
        label="step2:profile",
        system=PROFILE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [cv_block, {"type": "text", "text": "Extract this candidate's profile."}],
            }
        ],
    )


def profile_brief(p: CandidateProfile) -> str:
    """A compact text rendering of a profile, for putting into other prompts.

    Kept stable and deterministic on purpose: this string becomes a cached
    prompt prefix in scoring.py, and any wobble in it destroys the cache hit.
    """
    return (
        f"Name: {p.name}\n"
        f"Headline: {p.headline}\n"
        f"Seniority: {p.seniority} ({p.years_experience:g} years)\n"
        f"Core skills: {', '.join(p.core_skills)}\n"
        f"Domains: {', '.join(p.domains) or 'n/a'}\n"
        f"Target roles: {', '.join(p.target_roles)}\n"
        f"Languages: {', '.join(p.languages) or 'n/a'}\n"
        f"Locations: {', '.join(p.locations) or 'n/a'}\n"
        f"Open to remote: {'yes' if p.open_to_remote else 'no'}\n"
        f"Notable: " + "; ".join(p.notable)
    )
