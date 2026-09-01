"""Tests for the parts that quietly break.

    pip install pytest
    pytest -q

Not a comprehensive suite. Every test here exists because something actually
went wrong during the build, or because it locks in a decision that looks like
a bug and must not be "fixed" by a well-meaning student.

**None of these make an API call.** They run on the snapshot and the no-model
engine, so they are free, fast, and work with no key and no network. That is
deliberate: a test suite you cannot afford to run is not a test suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import engines  # noqa: E402
import jobs as job_data  # noqa: E402
import tools as tool_module  # noqa: E402
from engines.base import CVSource  # noqa: E402


# --- The data layer ----------------------------------------------------------


@pytest.fixture(scope="module")
def snapshot():
    if not config.SNAPSHOT_PATH.exists():
        pytest.skip("no snapshot — run: python build_snapshot.py")
    import json

    return json.loads(config.SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_snapshot_has_both_sources(snapshot):
    assert len(snapshot["israel"]) > 100
    assert len(snapshot["remote"]) > 100


def test_remote_jobs_are_deduplicated(snapshot):
    """The Himalayas feed serves overlapping pages. A naive 1500-job pull came
    back ~35% duplicated, so the same role appeared four times in one result."""
    ids = [j["id"] for j in snapshot["remote"]]
    assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} duplicate remote jobs"


def test_no_broken_company_names(snapshot):
    """~5% of Himalayas records carry the literal string "name" as the company.
    We recover it from the URL slug instead of showing a job at a company
    called "name" — which would also produce an applications/name-.../ folder."""
    broken = [j for j in snapshot["remote"] if j["company"].strip().lower() in ("name", "")]
    assert not broken, f"{len(broken)} jobs with an unusable company field"


def test_posted_dates_are_dates_not_epochs(snapshot):
    """`str(pubDate)[:10]` silently truncates a 10-digit epoch to itself, so
    every job claimed to be posted in 1787097505 and nothing errored."""
    for job in snapshot["remote"][:200]:
        if job["posted"]:
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", job["posted"]), \
                f"{job['posted']!r} is not a date"


# --- keyword_filter ----------------------------------------------------------


def test_keyword_filter_respects_word_boundaries():
    """Substring matching made "go" match Google, Logos, algorithm and mongodb,
    so a backend search returned SEO and sales roles ahead of real matches."""
    fake = [
        {"title": "Local SEO Manager (Google Business Profile)", "description": "marketing"},
        {"title": "Enterprise Account Executive - New Logos", "description": "sales quota"},
        {"title": "Go Backend Engineer", "description": "we use Go and Kubernetes"},
    ]
    for job in fake:
        job.setdefault("id", job["title"])

    hits = [j["title"] for j in job_data.keyword_filter(fake, "go", limit=10)]
    assert "Go Backend Engineer" in hits
    assert not any("SEO" in h or "Logos" in h for h in hits), \
        f"substring matching is back: {hits}"


def test_keyword_filter_ranks_title_matches_first():
    fake = [
        {"id": "a", "title": "Product Manager", "description": "we work with kubernetes daily"},
        {"id": "b", "title": "Kubernetes Platform Engineer", "description": "infrastructure"},
    ]
    assert job_data.keyword_filter(fake, "kubernetes", limit=2)[0]["id"] == "b"


# --- Deliberate failures that must NOT be fixed ------------------------------


def test_looks_israeli_still_fails_on_real_locations():
    """Step 4 exists to watch this lose to an LLM. If someone "fixes" it, the
    workshop's central comparison quietly stops working — so the failure is
    pinned here on purpose.

    Note what it gets right, because it matters for how you demo this: the list
    handles anything containing "Tel Aviv" or "Israel", which on a real board is
    most listings. The failures are bare city names and, memorably, the same
    city typed with a curly apostrophe instead of an ASCII one.
    """
    assert job_data.looks_israeli("Tel Aviv, Israel")
    assert job_data.looks_israeli("Giv'atayim")          # ASCII apostrophe: in the list

    must_miss = [
        "Giv‘atayim",   # U+2018 — the same city, typed the way a Mac types it
        "Ra’anana",     # U+2019 — likewise
        "Kiryat Gat", "Holon", "Bnei Brak", "Ashdod", "Airport City",
    ]
    still_missed = [loc for loc in must_miss if not job_data.looks_israeli(loc)]
    assert still_missed, ("looks_israeli() no longer misses anything — Step 4's "
                          "regex-vs-LLM demo needs it to fail. Do not fix it.")


def test_step4_bakeoff_has_material_to_show():
    """The demo needs locations the regex actually gets wrong.

    On the live snapshot it found zero disagreements, because Israeli postings
    redundantly include "Tel Aviv" or "Israel" alongside the city. So the hard
    cases are curated rather than sampled — otherwise Step 4 runs, finds
    nothing, and the point never lands."""
    import scoring

    assert getattr(scoring, "HARD_LOCATIONS", None), "Step 4 has no curated hard cases"
    missed = [loc for loc in scoring.HARD_LOCATIONS if not job_data.looks_israeli(loc)]
    assert len(missed) >= 5, f"only {len(missed)} of the curated cases actually fail"


def test_profile_brief_is_stable():
    """It becomes a cached prompt prefix in scoring.py. Any wobble in this
    string destroys the cache hit that Step 4 demonstrates."""
    from profile import profile_brief

    p = engines.get_engine("heuristic").extract_profile(
        CVSource.from_path(ROOT / "sample_cvs" / "senior_backend.pdf"))
    assert profile_brief(p) == profile_brief(p)
    assert profile_brief(p).startswith("Name: ")


# --- The approval gate -------------------------------------------------------


def test_agent_cannot_save_an_application():
    """The gate is a wire that was never connected, not a prompt instruction.
    If save_application ever appears in the tool list, the gate is gone."""
    names = {t["name"] for t in tool_module.TOOLS_WITH_APPLY}
    assert "draft_application" in names
    assert "save_application" not in names
    assert "save_application" not in tool_module.IMPLEMENTATIONS


def test_draft_never_reports_itself_as_sent():
    engine = engines.get_engine("heuristic")
    cv = CVSource.from_path(ROOT / "sample_cvs" / "senior_backend.pdf")
    profile = engine.extract_profile(cv)
    job = job_data.get_israel_jobs()[0]
    result = engine.draft(profile, job)
    assert result["status"] == "awaiting_approval"
    assert result.get("submitted") is not True


# --- The engine interface ----------------------------------------------------


@pytest.mark.parametrize("name", ["claude", "offline", "local", "heuristic"])
def test_every_engine_implements_the_interface(name):
    """The bug this catches: an engine that imports fine but is missing a method
    only explodes when a user clicks the button that needs it."""
    engine = engines.get_engine(name)
    for method in ("summarize_cv", "extract_profile", "run_agent", "rank", "draft", "available"):
        assert callable(getattr(engine, method, None)), f"{name} is missing {method}()"


def test_default_engine_is_always_runnable():
    """Whatever the machine has or lacks, the default must actually work."""
    engine = engines.get_engine(engines.default_engine_name())
    assert engine.available()[0]


# --- The no-model engine -----------------------------------------------------


@pytest.fixture(scope="module")
def heuristic():
    return engines.get_engine("heuristic")


def test_heuristic_reads_each_sample_cv(heuristic):
    for stem in ("senior_backend", "junior_frontend", "data_scientist"):
        p = heuristic.extract_profile(CVSource.from_path(ROOT / "sample_cvs" / f"{stem}.pdf"))
        assert p.name != "Unknown", f"{stem}: could not find a name"
        assert len(p.core_skills) >= 4, f"{stem}: only found {p.core_skills}"
        assert p.years_experience > 0


def test_heuristic_ranks_skills_by_evidence_not_vocabulary_order(heuristic):
    """The original bug: core_skills came back in SKILLS-list order and was
    truncated to 15, so a machine-learning CV lost PyTorch, RAG and LLM off the
    end and its keywords read "python go sql snowflake spark"."""
    p = heuristic.extract_profile(CVSource.from_path(ROOT / "sample_cvs" / "data_scientist.pdf"))
    assert any(s in p.core_skills[:6] for s in ("PyTorch", "LLM", "RAG")), \
        f"ML skills not surfaced: {p.core_skills[:6]}"


def test_heuristic_does_not_hallucinate_go_from_prose(heuristic):
    """"go to production" is not Go experience. Short ambiguous tokens are
    matched case-sensitively for exactly this reason."""
    p = heuristic.extract_profile(CVSource.from_path(ROOT / "sample_cvs" / "junior_frontend.pdf"))
    assert "Go" not in p.core_skills


def test_heuristic_does_not_promote_a_junior(heuristic):
    """Seniority used to come from any seniority word in the first 1200 chars,
    which read a two-year frontend developer as "senior"."""
    p = heuristic.extract_profile(CVSource.from_path(ROOT / "sample_cvs" / "junior_frontend.pdf"))
    assert p.seniority in ("junior", "mid"), f"got {p.seniority} for a 2-year CV"


def test_heuristic_full_pipeline_is_free_and_fast(heuristic):
    """The whole point of this engine: no key, no network, about a second."""
    tool_module.JOB_REGISTRY.clear()
    cv = CVSource.from_path(ROOT / "sample_cvs" / "senior_backend.pdf")
    profile = heuristic.extract_profile(cv)
    heuristic.run_agent(profile, search_modes=("Israel",))
    pool = list(tool_module.JOB_REGISTRY.values())[:10]
    assert pool, "no jobs found"
    ranked = heuristic.rank(profile, pool, top_n=5)
    assert all(m.location_ok for _, m in ranked)
    assert all(m.gaps for _, m in ranked), "every match must name at least one gap"


# --- The tool layer ----------------------------------------------------------


def test_tools_return_errors_instead_of_raising():
    """An agent told "that failed, here is why" can recover. One whose process
    crashed cannot."""
    out, is_error = tool_module.run_tool("no_such_tool", {})
    assert is_error and "No such tool" in out

    out, is_error = tool_module.run_tool("get_job_details", {"wrong_arg": 1})
    assert is_error and "Bad arguments" in out


def test_search_then_details_roundtrip():
    tool_module.JOB_REGISTRY.clear()
    import json

    out, is_error = tool_module.run_tool("search_israel_jobs", {"keywords": "backend", "limit": 3})
    assert not is_error
    results = json.loads(out)["results"]
    assert results

    out, is_error = tool_module.run_tool("get_job_details", {"job_id": results[0]["job_id"]})
    assert not is_error
    assert json.loads(out)["description"]


def test_unknown_job_id_is_rejected():
    import json

    out, _ = tool_module.run_tool("get_job_details", {"job_id": "made-up"})
    assert "error" in json.loads(out)


# --- The apply path ----------------------------------------------------------


def test_nothing_in_the_codebase_submits_an_application():
    """The whole design rests on this: no code path POSTs to a careers endpoint.

    If someone adds one, the approval gate stops being architectural and becomes
    a promise. Grep is a blunt instrument, and it is the right one here."""
    import re

    suspicious = []
    for path in ROOT.glob("*.py"):
        if path.name.startswith("test"):
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(requests\.post|httpx\.post|urlopen)\s*\(", text):
            line = text[:match.start()].count("\n") + 1
            suspicious.append(f"{path.name}:{line}")
    assert not suspicious, f"outbound POST found in the apply path: {suspicious}"


def test_apply_log_records_that_we_never_submitted():
    """Every logged row must assert submitted_by_this_tool is False."""
    import inspect

    import apply as apply_module

    source = inspect.getsource(apply_module.record_application)
    assert '"submitted_by_this_tool": False' in source
