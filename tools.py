"""Tool definitions and their implementations.

A tool is two things that must be kept in sync:

  1. A JSON schema telling Claude what the tool is for and what arguments it
     takes. The model only ever sees this. If the description is vague, the
     model calls the tool wrongly, and no amount of clever loop code saves you.
  2. A plain Python function that does the work. It knows nothing about Claude.

That is the whole idea. Notice how much of the effort below goes into the
*descriptions* rather than the code — writing tool descriptions is the actual
skill, and it is closer to writing documentation than to writing software.
"""

from __future__ import annotations

import json

import jobs as job_data

# Every job the agent has seen this session, keyed by id. Search results are
# deliberately short (title, company, location, one line), so the agent needs a
# way to pull the full text of the two or three it actually cares about. This
# registry is what makes get_job_details possible.
JOB_REGISTRY: dict[str, dict] = {}


def _register(found: list[dict]) -> None:
    for job in found:
        JOB_REGISTRY[job["id"]] = job


def _summarize(job: dict) -> dict:
    """The compact form of a job that goes back to the model.

    Sending 25 full job descriptions would be perhaps 200k tokens, most of it
    boilerplate about equal-opportunity employment and free snacks. Send the
    identifying fields and one line of substance; let the agent ask for more.
    Context is a budget, and search results are where people blow it.
    """
    excerpt = " ".join(job["description"].split())[:220]
    return {
        "job_id": job["id"],
        "title": job["title"],
        "company": job["company"],
        "location": job["location"],
        "salary": job["salary"] or "not stated",
        "excerpt": excerpt + ("..." if len(excerpt) == 220 else ""),
    }


# --- Implementations ---------------------------------------------------------


def search_israel_jobs(keywords: str = "", location: str = "", limit: int = 15,
                       live: bool = False) -> str:
    """Search open roles at Israeli tech companies, optionally by city.

    Both filters are optional. No keywords and no location returns the most
    recent roles, which is what you want when a candidate is exploring rather
    than hunting for one stack.
    """
    all_jobs = job_data.get_israel_jobs(live=live)
    israeli = [j for j in all_jobs if job_data.looks_israeli(j["location"])]
    pool = israeli

    if location:
        pool = job_data.filter_by_location(pool, location)
        if not pool:
            available = job_data.cities_present(israeli)
            return json.dumps({
                "results": [],
                "note": f"No roles found in '{location}'. Cities with openings right now: "
                        f"{', '.join(available) or 'none'}. Search without a location to see everything.",
            })

    found = job_data.keyword_filter(pool, keywords, limit=limit) if keywords else pool[:limit]
    _register(found)

    if not found:
        where = f" in {location}" if location else ""
        return json.dumps({
            "results": [],
            "note": f"No matches for '{keywords}'{where} among {len(pool)} roles. "
                    "Try broader keywords, or drop the location filter.",
        })

    return json.dumps(
        {
            "searched": len(pool),
            "filtered_by": {"keywords": keywords or None, "location": location or None},
            "returned": len(found),
            "results": [_summarize(j) for j in found],
        },
        ensure_ascii=False,
    )


def search_remote_jobs(keywords: str = "", limit: int = 15, live: bool = False) -> str:
    """Search remote roles. Keywords are optional — omit them to browse."""
    all_jobs = job_data.get_remote_jobs(live=live)
    found = job_data.keyword_filter(all_jobs, keywords, limit=limit) if keywords else all_jobs[:limit]
    _register(found)

    if not found:
        return json.dumps(
            {
                "results": [],
                "note": f"No matches for '{keywords}' among {len(all_jobs)} remote roles. "
                "Try broader or different keywords.",
            }
        )
    return json.dumps(
        {"searched": len(all_jobs), "returned": len(found), "results": [_summarize(j) for j in found]},
        ensure_ascii=False,
    )


def get_job_details(job_id: str) -> str:
    """Return the full posting for one job the agent has already found."""
    job = JOB_REGISTRY.get(job_id)
    if job is None:
        return json.dumps(
            {
                "error": f"Unknown job_id '{job_id}'. Only ids returned by a previous "
                "search can be looked up. Search first."
            }
        )

    # Trim very long postings. Some companies paste an entire handbook in here.
    description = job["description"]
    if len(description) > 6000:
        description = description[:6000] + "\n\n[...posting truncated...]"

    return json.dumps(
        {
            "job_id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "salary": job["salary"] or "not stated",
            "posted": job["posted"] or "unknown",
            "apply_url": job["url"],
            "description": description,
        },
        ensure_ascii=False,
    )


# --- Schemas -----------------------------------------------------------------
# Read these as the model reads them: this is the entire user manual it gets.

TOOLS = [
    {
        "name": "search_israel_jobs",
        "description": (
            "Search currently open roles at Israeli technology companies (Similarweb, "
            "Riskified, Lightricks, JFrog, Gong, Fireblocks, Orca Security, Taboola, "
            "Payoneer, Melio). Only returns positions physically based in Israel. "
            "Both filters are optional: pass keywords to search by technology, pass a "
            "location to restrict to one city, pass both, or neither to browse. Use this "
            "for a candidate who can work in Israel. Returns short summaries; call "
            "get_job_details for the full posting of any role that looks promising."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": (
                        "OPTIONAL. Space-separated search terms — technologies and role "
                        "words, lowercase, no punctuation. Example: 'backend go kubernetes'. "
                        "Fewer, more specific terms work better than a long list. Omit "
                        "entirely to browse everything, which is the right move when the "
                        "candidate is exploring or when a keyword search came back empty."
                    ),
                },
                "location": {
                    "type": "string",
                    "description": (
                        "OPTIONAL. Restrict to one Israeli city — for example 'Tel Aviv', "
                        "'Herzliya', 'Jerusalem', 'Haifa', 'Ra'anana', 'Be'er Sheva'. "
                        "Spelling variants are handled. Use it when the candidate has said "
                        "where they can commute to; leave it out otherwise, since most "
                        "listings are around Tel Aviv and a narrow filter can empty the "
                        "results. If a city has no openings the tool tells you which do."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "How many results to return, 1-25. Default 15.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_remote_jobs",
        "description": (
            "Search remote roles from a global board of about 100,000 listings. These "
            "are location-independent positions, though many carry a country or "
            "timezone restriction which is shown in the location field — check it "
            "before recommending a role. Use this for candidates open to remote work. "
            "Returns short summaries; call get_job_details for the full posting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": (
                        "OPTIONAL. Space-separated search terms — technologies and role "
                        "words, lowercase, no punctuation. Example: 'react typescript "
                        "frontend'. Omit to browse the most recent remote listings."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "How many results to return, 1-25. Default 15.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_job_details",
        "description": (
            "Fetch the complete posting for one job: full description, requirements, "
            "salary if stated, and the application URL. Only works for a job_id that a "
            "previous search returned. Use this on the handful of roles that look like "
            "a real fit — do not call it on every result, the full text is long."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "A job_id exactly as returned by one of the search tools.",
                }
            },
            "required": ["job_id"],
        },
    },
]

# name -> function, so the loop can dispatch without a chain of if-statements.
IMPLEMENTATIONS = {
    "search_israel_jobs": search_israel_jobs,
    "search_remote_jobs": search_remote_jobs,
    "get_job_details": get_job_details,
}


def run_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """Execute one tool call. Returns (result_text, is_error).

    Every exception is caught and handed back to the model as an error result
    rather than raised. An agent that gets told "that failed, here is why" can
    recover — try different arguments, try another tool, or explain the problem
    to the user. An agent whose process crashed can do none of those things.
    """
    func = IMPLEMENTATIONS.get(name)
    if func is None:
        return f"No such tool: {name}", True
    try:
        return func(**arguments), False
    except TypeError as exc:
        return f"Bad arguments for {name}: {exc}", True
    except Exception as exc:  # noqa: BLE001
        return f"{name} failed: {type(exc).__name__}: {exc}", True


# --- Step 6: the apply tool --------------------------------------------------
# Added separately so steps 3-5 run with three tools and step 6 adds a fourth.
# Pass TOOLS_WITH_APPLY to run_agent() when you want the agent to draft.

# The candidate is session state, not a tool argument. The model should not be
# passing a whole profile back to us on every call — it would cost tokens, and
# the model could quietly alter it in transit. Your application owns this.
CURRENT_PROFILE = None


def set_profile(profile) -> None:
    """Give the tool layer the candidate it is working for."""
    global CURRENT_PROFILE
    CURRENT_PROFILE = profile


def draft_application(job_id: str) -> str:
    """Draft an application for one job. Saves nothing — a human must approve."""
    if CURRENT_PROFILE is None:
        return json.dumps({"error": "No candidate profile loaded. Call set_profile() first."})

    job = JOB_REGISTRY.get(job_id)
    if job is None:
        return json.dumps(
            {"error": f"Unknown job_id '{job_id}'. Search for the job first, then draft."}
        )

    import apply as apply_module

    result = apply_module.draft_application(CURRENT_PROFILE, job)
    return json.dumps(result, ensure_ascii=False)


APPLY_TOOL = {
    "name": "draft_application",
    "description": (
        "Write a tailored cover letter and CV bullets for one job the candidate is "
        "interested in. Use it only on roles you have already recommended and that "
        "the candidate has a real chance at.\n\n"
        "This tool does NOT apply for the job. It produces a draft and returns "
        "status 'awaiting_approval'. Nothing is saved and nothing is submitted — a "
        "human reads the draft and decides. After calling it, show the candidate "
        "what you wrote and tell them it is waiting for their approval. Do not "
        "describe the application as sent, submitted, or filed, because it is none "
        "of those things."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "A job_id returned by an earlier search.",
            }
        },
        "required": ["job_id"],
    },
}

IMPLEMENTATIONS["draft_application"] = draft_application

TOOLS_WITH_APPLY = TOOLS + [APPLY_TOOL]
