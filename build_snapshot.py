"""Build data/jobs_snapshot.json — run this once before the workshop.

Why a snapshot:
  1. Thirty laptops hitting a public job board in the same 90 seconds is rude
     and slow, and some boards will start refusing.
  2. Your live demo becomes reproducible. "The agent found this job" still
     works when you rehearse it at midnight.
  3. It is half of offline mode. With the snapshot plus data/cache/, the whole
     workshop runs with no network at all.

Usage:
    python build_snapshot.py            # fetch everything, write the snapshot
    python build_snapshot.py --probe    # just check which boards are alive
"""

import json
import sys
from datetime import date

import requests

from config import SNAPSHOT_PATH
from data.boards import GREENHOUSE_BOARDS, GREENHOUSE_URL
from jobs import fetch_all_boards, fetch_all_remote, looks_israeli


def probe() -> None:
    """Check every configured board and report which ones still answer."""
    print("Probing company boards...\n")
    dead = []
    for token, company in GREENHOUSE_BOARDS:
        try:
            resp = requests.get(GREENHOUSE_URL.format(token=token), timeout=15)
            if resp.status_code == 200:
                count = len(resp.json().get("jobs", []))
                print(f"  OK    {company:<16} {count:>4} jobs   ({token})")
            else:
                print(f"  DEAD  {company:<16} HTTP {resp.status_code}   ({token})")
                dead.append(token)
        except Exception as exc:  # noqa: BLE001
            print(f"  DEAD  {company:<16} {type(exc).__name__}   ({token})")
            dead.append(token)

    if dead:
        print(f"\nRemove these from data/boards.py: {', '.join(dead)}")
    else:
        print("\nAll boards healthy.")


def build() -> None:
    print("Fetching company boards (Greenhouse, Lever, Ashby)...")
    israel_all = fetch_all_boards(with_content=True)
    israel_local = [j for j in israel_all if looks_israeli(j["location"])]

    print("\nFetching remote jobs (Himalayas)...")
    # 1500, not 300. Only ~2% of Himalayas listings are open to a candidate in
    # Israel — the rest carry a country restriction — so a 300-job pull leaves
    # about seven eligible roles and the remote half of the demo shows nothing.
    remote = fetch_all_remote(limit=1500)

    snapshot = {
        "built": date.today().isoformat(),
        "israel": israel_all,
        "remote": remote,
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")

    size_mb = SNAPSHOT_PATH.stat().st_size / 1_000_000
    print(
        f"\nWrote {SNAPSHOT_PATH.name}: "
        f"{len(israel_all)} company-board jobs ({len(israel_local)} of them in Israel), "
        f"{len(remote)} remote jobs, {size_mb:.1f} MB"
    )


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    else:
        build()
