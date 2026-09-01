"""Run this first. It tells you whether you are ready for the workshop.

    python setup_check.py

Checks, in the order that matters:
  1. Python version
  2. Dependencies installed
  3. Which credential path you are on — your own key, a shared key, or offline
  4. The job data (snapshot present, or the boards reachable)
  5. One real round-trip to the API, if a key is available

Every check prints PASS, WARN or FAIL and keeps going, so you get the whole
picture in one run rather than fixing one thing at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(status: str, title: str, detail: str = "") -> None:
    _results.append((status, title, detail))
    mark = {OK: "\033[32m PASS \033[0m", WARN: "\033[33m WARN \033[0m", FAIL: "\033[31m FAIL \033[0m"}[status]
    print(f"[{mark}] {title}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def main() -> int:
    print("\nJob Agent Workshop — setup check\n" + "=" * 50)

    # 1. Python -------------------------------------------------------------
    v = sys.version_info
    if v >= (3, 10):
        check(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        check(FAIL, f"Python {v.major}.{v.minor} is too old", "This workshop needs Python 3.10 or newer.")
        return 1

    # 2. Dependencies -------------------------------------------------------
    missing = []
    for module, package in [
        ("anthropic", "anthropic"),
        ("streamlit", "streamlit"),
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        ("reportlab", "reportlab"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        check(FAIL, "Dependencies missing", f"pip install {' '.join(missing)}")
        return 1
    import anthropic

    check(OK, "Dependencies installed", f"anthropic {anthropic.__version__}")

    # 3. Credentials --------------------------------------------------------
    import config

    if config.OFFLINE:
        cached = len(list(config.CACHE_DIR.glob("*.json"))) if config.CACHE_DIR.exists() else 0
        if cached:
            check(OK, "OFFLINE=1 — running from recorded responses", f"{cached} responses in data/cache/")
        else:
            check(FAIL, "OFFLINE=1 but data/cache/ is empty",
                  "Someone needs to run the recorder once with a key:\n"
                  "    RECORD=1 python record_cache.py")
    elif config.have_api_key():
        check(OK, "API key found", f"model: {config.MODEL}")
    else:
        check(FAIL, "No API key and OFFLINE is not set",
              "Either:  cp .env.example .env   and paste a key\n"
              "Or:      echo 'OFFLINE=1' >> .env")

    # 4. Job data -----------------------------------------------------------
    if config.SNAPSHOT_PATH.exists():
        import json

        snap = json.loads(config.SNAPSHOT_PATH.read_text(encoding="utf-8"))
        import jobs as job_data

        israel_local = [j for j in snap["israel"] if job_data.looks_israeli(j["location"])]
        check(OK, "Job snapshot present",
              f"built {snap.get('built', '?')} — {len(snap['israel'])} company-board jobs "
              f"({len(israel_local)} matched as Israel-based), {len(snap['remote'])} remote")
    else:
        check(WARN, "No job snapshot", "Run:  python build_snapshot.py")

    # 5. Sample CVs ---------------------------------------------------------
    cvs = sorted(config.SAMPLE_CV_DIR.glob("*.pdf")) if config.SAMPLE_CV_DIR.exists() else []
    if cvs:
        check(OK, f"{len(cvs)} sample CVs", ", ".join(p.name for p in cvs))
    else:
        check(WARN, "No sample CVs", "Run:  python make_sample_cvs.py")

    # 6. A real call --------------------------------------------------------
    if config.OFFLINE:
        check(WARN, "Skipped the live API check", "OFFLINE=1 — nothing to call.")
    elif not config.have_api_key():
        check(WARN, "Skipped the live API check", "No key to call with.")
    else:
        try:
            import llm

            response = llm.create(
                messages=[{"role": "user", "content": "Reply with exactly: ready"}],
                max_tokens=16,
            )
            check(OK, "API round-trip works", f"model replied: {llm.text_of(response)!r}")
        except anthropic.AuthenticationError:
            check(FAIL, "API key was rejected", "Check the key in your .env file.")
        except anthropic.APIConnectionError:
            check(FAIL, "Could not reach the API", "Network or proxy problem. OFFLINE=1 is your fallback.")
        except Exception as exc:  # noqa: BLE001
            check(FAIL, f"API call failed: {type(exc).__name__}", str(exc)[:300])

    # Summary ---------------------------------------------------------------
    print("=" * 50)
    fails = sum(1 for s, _, _ in _results if s == FAIL)
    warns = sum(1 for s, _, _ in _results if s == WARN)
    if fails:
        print(f"\n{fails} blocking problem(s). Fix those before the workshop starts.\n")
        return 1
    print(f"\nReady to go{f' ({warns} warning(s))' if warns else ''}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
