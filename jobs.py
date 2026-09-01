"""The data layer: fetch and normalize job listings from public boards.

This module is deliberately boring. It has no LLM calls in it at all — it just
turns two different JSON shapes into one dict shape. Keeping the "get the data"
code separate from the "reason about the data" code is the point.

A normalized job looks like:

    {
      "id":          "gh-lightricks-1234567",
      "title":       "Senior Backend Engineer",
      "company":     "Lightricks",
      "location":    "Tel Aviv/ Netanya, Israel",   # raw, messy, as published
      "source":      "greenhouse" | "himalayas",
      "url":         "https://...",                  # where a human applies
      "description": "plain text, HTML stripped",
      "employment_type": "Full Time" | "",
      "seniority":   ["Senior"],
      "salary":      "USD 120000-160000/annual" | "",
      "posted":      "2026-08-01" | "",
    }
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import requests

from config import HTTP_TIMEOUT, SNAPSHOT_PATH, USE_SNAPSHOT
from data.boards import (
    ASHBY_BOARDS,
    ASHBY_URL,
    GREENHOUSE_BOARDS,
    GREENHOUSE_URL,
    HIMALAYAS_URL,
    ISRAEL_HINTS,
    JOBICY_URL,
    LEVER_BOARDS,
    LEVER_URL,
    REMOTIVE_URL,
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def strip_html(raw: str) -> str:
    """Turn a job board's HTML blob into readable plain text.

    Job descriptions arrive as HTML-escaped HTML. We unescape once, drop the
    tags, and collapse whitespace. Block-level tags become newlines so the
    result keeps its bullet structure — that structure is signal for the model.
    """
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<li[^>]*>", "\n- ", text, flags=re.I)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)  # entities can survive one pass
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def looks_israeli(location: str) -> bool:
    """Naive substring check against a hand-written list of Israeli places.

    Step 4 of the workshop pits this function against an LLM. It loses. That is
    the entire reason it exists — do not "fix" it.
    """
    low = (location or "").lower()
    return any(hint in low for hint in ISRAEL_HINTS)


# --- Greenhouse (Israeli companies) ------------------------------------------


def fetch_greenhouse_board(token: str, company: str, with_content: bool = True) -> list[dict]:
    """Fetch one company's public Greenhouse board."""
    url = GREENHOUSE_URL.format(token=token)
    params = {"content": "true"} if with_content else {}
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    jobs = []
    for job in payload.get("jobs", []):
        jobs.append(
            {
                "id": f"gh-{token}-{job.get('id')}",
                "title": job.get("title", "").strip(),
                "company": company,
                "location": (job.get("location") or {}).get("name", "").strip(),
                "source": "greenhouse",
                "url": job.get("absolute_url", ""),
                "description": strip_html(job.get("content", "")),
                "employment_type": "",
                "seniority": [],
                "salary": "",
                "posted": (job.get("first_published") or "")[:10],
            }
        )
    return jobs


def fetch_all_greenhouse(with_content: bool = True) -> list[dict]:
    """Fetch every configured Israeli company board, skipping any that fail.

    One dead board must not take down the whole workshop, so failures are
    swallowed with a note rather than raised.
    """
    all_jobs = []
    for token, company in GREENHOUSE_BOARDS:
        try:
            found = fetch_greenhouse_board(token, company, with_content)
            all_jobs.extend(found)
            print(f"  {company:<16} {len(found):>4} jobs")
        except Exception as exc:  # noqa: BLE001 - board health is not our concern
            print(f"  {company:<16}  !! skipped ({type(exc).__name__}: {exc})")
    return all_jobs


# --- Lever and Ashby ---------------------------------------------------------
# Same job, three vocabularies. Each provider names the fields differently and
# dates them differently, which is why normalising into one dict shape early —
# rather than teaching the rest of the app three schemas — is worth the work.


def fetch_lever(token: str, company: str) -> list[dict]:
    """Fetch one Lever board. `createdAt` is epoch milliseconds."""
    resp = requests.get(LEVER_URL.format(token=token), timeout=HTTP_TIMEOUT)
    resp.raise_for_status()

    jobs = []
    for job in resp.json():
        categories = job.get("categories") or {}
        posted = ""
        if job.get("createdAt"):
            try:
                posted = datetime.fromtimestamp(
                    int(job["createdAt"]) / 1000, tz=timezone.utc).date().isoformat()
            except (ValueError, OSError, OverflowError):
                posted = ""
        jobs.append({
            "id": f"lv-{token}-{job.get('id')}",
            "title": (job.get("text") or "").strip(),
            "company": company,
            "location": (categories.get("location") or "").strip(),
            "source": "lever",
            "url": job.get("hostedUrl") or job.get("applyUrl") or "",
            "description": strip_html(job.get("descriptionPlain")
                                      or job.get("description") or ""),
            "employment_type": categories.get("commitment") or "",
            "seniority": [],
            "salary": "",
            "posted": posted,
        })
    return jobs


def fetch_ashby(token: str, company: str) -> list[dict]:
    """Fetch one Ashby board.

    Ashby location strings are short and idiosyncratic — "TLV" for Tel Aviv,
    for instance — which is a good reminder that the hand-written place list in
    boards.py can never be finished.
    """
    resp = requests.get(ASHBY_URL.format(token=token), timeout=HTTP_TIMEOUT)
    resp.raise_for_status()

    jobs = []
    for job in resp.json().get("jobs", []):
        location = (job.get("location") or "").strip()
        if job.get("isRemote") and "remote" not in location.lower():
            location = f"Remote — {location}" if location else "Remote"
        jobs.append({
            "id": f"ash-{token}-{job.get('id')}",
            "title": (job.get("title") or "").strip(),
            "company": company,
            "location": location,
            "source": "ashby",
            "url": job.get("jobUrl") or job.get("applyUrl") or "",
            "description": strip_html(job.get("descriptionPlain")
                                      or job.get("descriptionHtml") or ""),
            "employment_type": job.get("employmentType") or "",
            "seniority": [],
            "salary": "",
            "posted": (job.get("publishedAt") or "")[:10],
        })
    return jobs


def fetch_all_boards(with_content: bool = True) -> list[dict]:
    """Every configured company board, across all three providers.

    A dead board must never take the whole workshop down, so each failure is
    reported and skipped rather than raised.
    """
    all_jobs = fetch_all_greenhouse(with_content)

    for fetch, boards, label in (
        (fetch_lever, LEVER_BOARDS, "lever"),
        (fetch_ashby, ASHBY_BOARDS, "ashby"),
    ):
        for token, company in boards:
            try:
                found = fetch(token, company)
                all_jobs.extend(found)
                print(f"  {company:<16} {len(found):>4} jobs   ({label})")
            except Exception as exc:  # noqa: BLE001
                print(f"  {company:<16}  !! skipped ({type(exc).__name__}: {exc})")
    return all_jobs


# --- Himalayas (remote) ------------------------------------------------------


def _himalayas_company(job: dict) -> str:
    """Recover the company name.

    About 5% of records come back with the literal string "name" in
    `companyName` — a bug on their side, not ours. The real name is sitting in
    the URL slug (/companies/atalnt-llc/jobs/...), so use that rather than
    showing the candidate a job at a company called "name".
    """
    name = (job.get("companyName") or "").strip()
    if name and name.lower() != "name":
        return name
    match = re.search(r"/companies/([^/]+)/", job.get("applicationLink") or "")
    return match.group(1).replace("-", " ").title() if match else "Unknown"


def _himalayas_date(raw) -> str:
    """Normalize pubDate to YYYY-MM-DD.

    It arrives as a Unix epoch. The obvious `str(raw)[:10]` looks like it works
    — a 10-digit epoch survives a 10-character truncation unchanged — so the
    bug ships silently and every job claims to be posted in 1787097505.
    """
    if not raw:
        return ""
    text = str(raw).strip()
    if text.isdigit():
        try:
            return datetime.fromtimestamp(int(text[:10]), tz=timezone.utc).date().isoformat()
        except (ValueError, OSError, OverflowError):
            return ""
    return text[:10]


def fetch_himalayas(limit: int = 200, offset: int = 0) -> list[dict]:
    """Fetch remote jobs from the Himalayas public API.

    Two things this API does that will bite you, both learned the hard way:

    1. It silently caps `limit` at 20 per call no matter what you ask for, so we
       page until we have enough. Asking for 100 and assuming you got 100 is the
       bug you would otherwise ship.
    2. Its pages overlap. Ask for 1500 naively and roughly a third come back
       duplicated — the same guid four or five times over. So we dedupe while
       paging, advance the offset by pages *requested* rather than rows kept,
       and give up after a few consecutive pages that contain nothing new.

    The general lesson: paginated APIs are not a guarantee of distinct results,
    and a set of ids costs you one line.
    """
    collected = []
    seen: set[str] = set()
    page_size = 20
    page = 0
    stale_pages = 0

    while len(collected) < limit:
        resp = requests.get(
            HIMALAYAS_URL,
            params={"limit": page_size, "offset": offset + page * page_size},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        batch = resp.json().get("jobs", [])
        page += 1

        if not batch:
            break

        fresh = [j for j in batch if j.get("guid") and j["guid"] not in seen]
        seen.update(j["guid"] for j in fresh)
        collected.extend(fresh)

        # A page of pure repeats is normal here; several in a row means we have
        # reached the end of what the feed will actually give us.
        stale_pages = stale_pages + 1 if not fresh else 0
        if stale_pages >= 5:
            break
        if len(batch) < page_size:
            break

    collected = collected[:limit]

    jobs = []
    for job in collected:
        salary = ""
        if job.get("minSalary") and job.get("maxSalary"):
            cur = job.get("currency") or ""
            per = job.get("salaryPeriod") or ""
            salary = f"{cur} {int(job['minSalary'])}-{int(job['maxSalary'])}/{per}".strip()

        jobs.append(
            {
                "id": f"him-{job.get('guid', '')}",
                "title": (job.get("title") or "").strip(),
                "company": _himalayas_company(job),
                "location": "Remote — " + ", ".join(job.get("locationRestrictions") or ["Worldwide"]),
                "source": "himalayas",
                "url": job.get("applicationLink", ""),
                "description": strip_html(job.get("description") or job.get("excerpt") or ""),
                "employment_type": job.get("employmentType") or "",
                "seniority": job.get("seniority") or [],
                "salary": salary,
                "posted": _himalayas_date(job.get("pubDate")),
            }
        )
    return jobs


def fetch_jobicy(count: int = 200) -> list[dict]:
    """Remote jobs from Jobicy. `jobGeo` is who may apply, e.g. "Anywhere"."""
    resp = requests.get(JOBICY_URL.format(count=count), timeout=HTTP_TIMEOUT,
                        headers={"User-Agent": "job-agent-workshop/1.0"})
    resp.raise_for_status()

    jobs = []
    for job in resp.json().get("jobs", []):
        geo = (job.get("jobGeo") or "Anywhere").strip()
        jobs.append({
            "id": f"jby-{job.get('id')}",
            "title": (job.get("jobTitle") or "").strip(),
            "company": (job.get("companyName") or "").strip() or "Unknown",
            "location": f"Remote — {geo}",
            "source": "jobicy",
            "url": job.get("url") or "",
            "description": strip_html(job.get("jobDescription")
                                      or job.get("jobExcerpt") or ""),
            "employment_type": ", ".join(job.get("jobType") or []),
            "seniority": [job["jobLevel"]] if job.get("jobLevel") else [],
            "salary": "",
            "posted": (job.get("pubDate") or "")[:10],
        })
    return jobs


def fetch_remotive() -> list[dict]:
    """Remote jobs from Remotive. Small feed, but most of it is worldwide."""
    resp = requests.get(REMOTIVE_URL, timeout=HTTP_TIMEOUT,
                        headers={"User-Agent": "job-agent-workshop/1.0"})
    resp.raise_for_status()

    jobs = []
    for job in resp.json().get("jobs", []):
        where = (job.get("candidate_required_location") or "Worldwide").strip()
        jobs.append({
            "id": f"rmt-{job.get('id')}",
            "title": (job.get("title") or "").strip(),
            "company": (job.get("company_name") or "").strip() or "Unknown",
            "location": f"Remote — {where}",
            "source": "remotive",
            "url": job.get("url") or "",
            "description": strip_html(job.get("description") or ""),
            "employment_type": job.get("job_type") or "",
            "seniority": [],
            "salary": job.get("salary") or "",
            "posted": (job.get("publication_date") or "")[:10],
        })
    return jobs


def fetch_all_remote(limit: int = 1500) -> list[dict]:
    """Every remote feed. A dead one is skipped, never fatal."""
    jobs = []
    for label, call in (("Himalayas", lambda: fetch_himalayas(limit=limit)),
                        ("Jobicy", lambda: fetch_jobicy(200)),
                        ("Remotive", fetch_remotive)):
        try:
            found = call()
            jobs.extend(found)
            print(f"  {label:<16} {len(found):>4} jobs")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<16}  !! skipped ({type(exc).__name__}: {exc})")

    seen, unique = set(), []
    for job in jobs:
        if job["id"] not in seen:
            seen.add(job["id"])
            unique.append(job)
    return unique


# --- Snapshot ----------------------------------------------------------------


@lru_cache(maxsize=4)
def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict:
    """Read the snapshot once, not once per search.

    Without the cache this re-reads and re-parses 11 MB of JSON on every single
    tool call. The agent makes four or five searches per run, the scorer asks
    again, and the whole test suite spent most of its time here.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No job snapshot at {path}.\n"
            "Run:  python build_snapshot.py\n"
            "(or pass live=True to fetch from the boards directly)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_israel_jobs(live: bool = False) -> list[dict]:
    """All jobs from the Israeli company boards (not yet filtered by location)."""
    if live or not USE_SNAPSHOT:
        return fetch_all_boards()
    return load_snapshot()["israel"]


def get_remote_jobs(live: bool = False, limit: int = 1500) -> list[dict]:
    """Remote jobs.

    The ceiling is high because most remote listings are restricted to a country
    the candidate cannot work from. Filtering happens later, so the pool handed
    to keyword_filter has to be big enough to survive it.
    """
    if live or not USE_SNAPSHOT:
        return fetch_all_remote(limit=limit)
    return load_snapshot()["remote"][:limit]


def keyword_filter(jobs: list[dict], keywords: str, limit: int = 25) -> list[dict]:
    """Rank jobs by how many of the given keywords appear in title + description.

    A plain scoring function, no LLM. It is a coarse pre-filter whose only job
    is to get the candidate pool down to something we can afford to have the
    model read carefully. Title matches are weighted heavily because a title is
    a much stronger signal than a passing mention in a benefits paragraph.

    Match on word boundaries, not substrings. `"go" in title` is true for
    "Google", "Logos", "algorithm" and "mongodb", so a plain `in` check hands a
    Go engineer a list of SEO and sales roles and buries the real matches. Short
    terms are common in this domain — go, ai, ml, r, c — and every one of them
    has the same problem.
    """
    terms = [t.strip().lower() for t in re.split(r"[,\s]+", keywords or "") if len(t.strip()) > 1]
    if not terms:
        return jobs[:limit]

    # Compiled once, not once per job: this runs over every posting in the pool.
    patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

    scored = []
    for job in jobs:
        title = job["title"].lower()
        body = job["description"].lower()[:4000]
        score = sum(8 for p in patterns if p.search(title)) + sum(1 for p in patterns if p.search(body))
        if score:
            scored.append((score, job))

    scored.sort(key=lambda pair: -pair[0])
    return [job for _, job in scored[:limit]]

# --- Filtering by place, inside Israel ---------------------------------------

# One entry per city, with the spellings that actually appear on job boards.
# Note the apostrophes: a board may emit Giv'atayim, Giv‘atayim or Givatayim for
# the same place, and matching only one of them silently drops listings. The
# normaliser below folds all three together.
CITY_ALIASES: dict[str, list[str]] = {
    "Tel Aviv": ["tel aviv", "tel-aviv", "telaviv", "tel aviv-yafo", "tlv", "yafo", "jaffa"],
    "Herzliya": ["herzliya", "herzeliya", "hertzliya"],
    "Jerusalem": ["jerusalem", "yerushalayim"],
    "Haifa": ["haifa", "hefa"],
    "Netanya": ["netanya", "nathanya"],
    "Ra'anana": ["ra'anana", "raanana"],
    "Petah Tikva": ["petah tikva", "petach tikva", "petah tiqwa"],
    "Ramat Gan": ["ramat gan", "ramat-gan"],
    "Giv'atayim": ["giv'atayim", "givatayim"],
    "Be'er Sheva": ["be'er sheva", "beer sheva", "beersheba"],
    "Rehovot": ["rehovot"],
    "Yokneam": ["yokneam", "yoqneam"],
    "Hod Hasharon": ["hod hasharon", "hod ha'sharon"],
    "Kfar Saba": ["kfar saba", "kfar sava"],
    "Caesarea": ["caesarea", "qesarya"],
    "Modi'in": ["modi'in", "modiin"],
    "Holon": ["holon"],
    "Bnei Brak": ["bnei brak", "bene beraq"],
}

_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'", "`": "'"})


def normalize_place(text: str) -> str:
    """Lowercase and fold the apostrophe variants boards use interchangeably."""
    return (text or "").translate(_APOSTROPHES).lower()


def known_cities() -> list[str]:
    """The canonical city names a search can ask for."""
    return sorted(CITY_ALIASES)


def cities_present(jobs: list[dict]) -> list[str]:
    """Which of the known cities actually appear in this batch of jobs.

    Worth calling before offering a filter to a user: the board set is small and
    heavily Tel Aviv, so most cities in the alias table match nothing today.
    """
    found = []
    for city, aliases in CITY_ALIASES.items():
        if any(any(a in normalize_place(j["location"]) for a in aliases) for j in jobs):
            found.append(city)
    return found


def filter_by_location(jobs: list[dict], location: str) -> list[dict]:
    """Keep jobs whose location matches the requested place.

    Accepts a canonical city name, any of its known spellings, or a free-text
    fragment. An unrecognised place falls back to a plain substring match rather
    than returning nothing, because a caller asking for somewhere we have never
    heard of is better served by a best effort than by silence.
    """
    wanted = normalize_place(location).strip()
    if not wanted:
        return jobs

    aliases = None
    for city, spellings in CITY_ALIASES.items():
        if wanted == normalize_place(city) or wanted in spellings:
            aliases = spellings
            break
    targets = aliases or [wanted]

    return [j for j in jobs if any(t in normalize_place(j["location"]) for t in targets)]
