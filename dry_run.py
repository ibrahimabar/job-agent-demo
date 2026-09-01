"""Dry run: the whole pipeline with no API key, no network, and no spend.

    python dry_run.py [cv_name]

For instructors rehearsing before a room, and for students whose key has not
arrived. It runs the real data path end to end on the no-model engine, so you
find out that a job board has gone dark BEFORE you are standing in front of
thirty people.

What is real here: the snapshot, both search tools, the keyword pre-filter, and
the location gate. What is not: the judgment. The no-model engine counts words —
see engines/heuristic.py for exactly how far that gets you, measured.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engines  # noqa: E402
import tools as tool_module  # noqa: E402


def main() -> int:
    stem = sys.argv[1] if len(sys.argv) > 1 else "senior_backend"
    cv_path = Path("sample_cvs") / f"{stem}.pdf"
    if not cv_path.exists():
        print(f"No CV at {cv_path}. Options: "
              f"{', '.join(p.stem for p in sorted(Path('sample_cvs').glob('*.pdf')))}")
        return 1

    engine = engines.get_engine("heuristic")
    print("=" * 74)
    print(f"DRY RUN — {engine.label}. No key, no network, no spend.")
    print("=" * 74)

    started = time.time()
    cv = engines.CVSource.from_path(cv_path)
    tool_module.JOB_REGISTRY.clear()

    profile = engine.extract_profile(cv)
    print(f"\n  {profile.name} — {profile.headline}")
    print(f"  keywords: {profile.search_keywords!r}")
    print(f"  can work from: {', '.join(profile.locations)}\n")

    print("-" * 74)
    print("SEARCH (real — this is the shipped tool layer)")
    print("-" * 74)
    engine.run_agent(profile, on_event=lambda kind, p: (
        print(f"  → {p['name']}({p['input'].get('keywords','')!r})")
        if kind == "tool_call" else None))

    pool = list(tool_module.JOB_REGISTRY.values())
    print(f"\n  {len(pool)} distinct roles in the registry\n")

    print("-" * 74)
    print("MATCHING (counted, not read)")
    print("-" * 74 + "\n")
    ranked = engine.rank(profile, pool[:25], top_n=6)
    print(f"  {len(ranked)} survived the location gate\n")
    for job, match in ranked:
        print(f"  {match.score:>3}  {job['title'][:52]}")
        print(f"       {job['company']} · {job['location'][:40]}")
        print(f"       fits: {match.why_you_fit[0] if match.why_you_fit else '-'}")
        print(f"       gaps: {match.gaps[0] if match.gaps else '-'}")
        if match.red_flags:
            print(f"       ! {match.red_flags[0]}")
        print()

    print("=" * 74)
    print(f"Done in {time.time() - started:.1f}s, $0.00 spent.")
    print("""
Measured against Claude ranking the same jobs, this engine scores Spearman
rho = +0.42. Better than chance, nowhere near reading. It cannot tell a hard
requirement from a nice-to-have, which is the one thing a candidate needs.
Run the same CV through the Claude engine to see the difference.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
