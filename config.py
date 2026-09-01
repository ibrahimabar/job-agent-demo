"""Central configuration for the workshop.

Everything that an instructor might want to change lives here, so students
never have to hunt through six files to swap a model or turn on offline mode.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOT_PATH = DATA_DIR / "jobs_snapshot.json"
SAMPLE_CV_DIR = ROOT / "sample_cvs"
APPLICATIONS_DIR = ROOT / "applications"

# --- Model -------------------------------------------------------------------
# One constant, used everywhere. Instructors running a shared key for a big
# room may prefer "claude-haiku-4-5" to cap spend; quality of the match
# explanations drops noticeably, so that is a deliberate trade, not a default.
MODEL = os.environ.get("WORKSHOP_MODEL", "claude-opus-5")

# Generous but not reckless. Non-streaming requests stay under SDK HTTP timeouts.
MAX_TOKENS = 16000

# --- Offline mode ------------------------------------------------------------
# OFFLINE=1 replays recorded Claude responses from data/cache/ and reads jobs
# from data/jobs_snapshot.json. Every step of the workshop completes with no
# API key and no network. This is the insurance policy for conference wifi.
OFFLINE = os.environ.get("OFFLINE", "").strip() in ("1", "true", "True", "yes")

# --- Public deployment -------------------------------------------------------
# PUBLIC=1 restricts the app to the engine that needs no key, no network and no
# GPU. Set it on a hosted copy so visitors cannot be prompted for an API key and
# so nothing they upload ever leaves the server.
PUBLIC = os.environ.get("PUBLIC", "").strip() in ("1", "true", "True", "yes")

# --- Engine picker -----------------------------------------------------------
# The sidebar shows all four engines so people can see what exists, but only the
# default is selectable. Unlock the picker when the paid and local engines are
# ready to be handed to someone else:
#     UNLOCK_ENGINES=1 streamlit run app.py
# The ENGINE variable still overrides regardless, so the command line is never
# locked out.
UNLOCK_ENGINES = os.environ.get("UNLOCK_ENGINES", "").strip() in ("1", "true", "True", "yes")

# --- Job sources -------------------------------------------------------------
# Use the pre-fetched snapshot by default: 30 laptops hammering a public job
# board at the same moment is both rude and slow, and the snapshot makes the
# instructor's demo reproducible. Pass --live to any script to override.
USE_SNAPSHOT = os.environ.get("USE_SNAPSHOT", "1").strip() in ("1", "true", "True", "yes")

HTTP_TIMEOUT = 20  # seconds, per request to a job board


def load_dotenv(path: Path = None) -> None:
    """Minimal .env loader so students don't need a python-dotenv install.

    Only sets variables that are not already present in the environment, so a
    real exported ANTHROPIC_API_KEY always wins over a shared workshop key.
    """
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

# Re-read after loading .env, in case OFFLINE / WORKSHOP_MODEL were set there.
MODEL = os.environ.get("WORKSHOP_MODEL", MODEL)
OFFLINE = OFFLINE or os.environ.get("OFFLINE", "").strip() in ("1", "true", "True", "yes")
PUBLIC = PUBLIC or os.environ.get("PUBLIC", "").strip() in ("1", "true", "True", "yes")
UNLOCK_ENGINES = UNLOCK_ENGINES or os.environ.get("UNLOCK_ENGINES", "").strip() in ("1", "true", "True", "yes")


def have_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
