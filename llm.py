"""Thin wrapper around the Anthropic client, plus the offline replay cache.

Students touch `client()` and `create()` and nothing else in here. The caching
machinery exists so that a room full of people can finish the workshop when the
wifi dies or the shared key hits a rate limit.

Three modes:
  live      (default)   call the API, and if RECORD=1 also save the response
  record    RECORD=1     live + write every response into data/cache/
  offline   OFFLINE=1    never touch the network; replay from data/cache/
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import anthropic

import costs
from config import CACHE_DIR, MAX_TOKENS, MODEL, OFFLINE, have_api_key

RECORD = os.environ.get("RECORD", "").strip() in ("1", "true", "True", "yes")

# RECORD normally re-calls the API for everything. With RESUME=1 (the default
# under RECORD) an already-cached request is served from disk instead, so a run
# that dies at call 90 of 120 does not make you pay for the first 89 again.
# Set RESUME=0 to force a full re-record after changing a prompt.
RESUME = os.environ.get("RESUME", "1").strip() in ("1", "true", "True", "yes")

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    """The shared Anthropic client.

    The zero-argument constructor resolves credentials from the environment —
    ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile.
    Never hardcode a key.
    """
    global _client
    if _client is None:
        if not have_api_key():
            print(
                "No ANTHROPIC_API_KEY found.\n"
                "  - Copy .env.example to .env and paste the workshop key, or\n"
                "  - run with OFFLINE=1 to use the recorded responses.",
                file=sys.stderr,
            )
        _client = anthropic.Anthropic()
    return _client


def _cache_key(**kwargs) -> str:
    """A stable fingerprint of a request.

    sort_keys matters: dict ordering must not change the hash, or the cache
    misses on every run for no visible reason.
    """
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def create(*, messages, system=None, tools=None, model=None, max_tokens=None, label=None, **extra):
    """`client.messages.create()` with offline replay bolted on.

    Everything about the call is ordinary — this returns exactly the Message
    object the SDK would return. The only addition is that the response can be
    served from, or written to, data/cache/.
    """
    model = model or MODEL
    max_tokens = max_tokens or MAX_TOKENS

    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        payload["system"] = system
    if tools is not None:
        payload["tools"] = tools
    payload.update(extra)

    key = _cache_key(**payload)
    cache_file = CACHE_DIR / f"{key}.json"

    if OFFLINE:
        if not cache_file.exists():
            raise RuntimeError(
                f"OFFLINE=1 but no recorded response for this request ({key}).\n"
                "The cache only covers the workshop's scripted path. If you changed a\n"
                "prompt or a CV, you need a live API key — or re-run the recorder:\n"
                "    RECORD=1 python record_cache.py"
            )
        return anthropic.types.Message.model_validate_json(cache_file.read_text("utf-8"))

    if RECORD and RESUME and cache_file.exists():
        return anthropic.types.Message.model_validate_json(cache_file.read_text("utf-8"))

    response = client().messages.create(**payload)
    costs.record(model, response.usage, label=label or "create")

    if RECORD:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(response.to_json(), encoding="utf-8")

    return response


def text_of(response) -> str:
    """Concatenate the text blocks of a response.

    A response is a *list* of content blocks — text, thinking, tool_use. Reaching
    for `response.content[0].text` works right up until the first time the model
    thinks or calls a tool, and then it throws. Always filter by type.
    """
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def parse(output_format, *, messages, system=None, model=None, max_tokens=None, label=None, **extra):
    """`client.messages.parse()` with the same offline replay.

    Returns a validated instance of `output_format` (a Pydantic model). The SDK
    constrains the model's output to your schema and validates it for you, so
    there is no "please return JSON" prompt-begging and no json.loads() that
    explodes on a stray markdown fence.

    The cache stores the *parsed* object rather than the raw Message, since the
    parsed object is the only thing callers want.
    """
    model = model or MODEL
    max_tokens = max_tokens or MAX_TOKENS

    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        payload["system"] = system
    payload.update(extra)

    key = _cache_key(schema=output_format.__name__, **payload)
    cache_file = CACHE_DIR / f"parse-{key}.json"

    if OFFLINE:
        if not cache_file.exists():
            raise RuntimeError(
                f"OFFLINE=1 but no recorded response for this request (parse-{key}).\n"
                "Re-run the recorder with a live key:  RECORD=1 python record_cache.py"
            )
        return output_format.model_validate_json(cache_file.read_text("utf-8"))

    if RECORD and RESUME and cache_file.exists():
        return output_format.model_validate_json(cache_file.read_text("utf-8"))

    response = client().messages.parse(output_format=output_format, **payload)
    result = response.parsed_output
    costs.record(model, response.usage, label=label or f"parse:{output_format.__name__}")

    if RECORD:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(result.model_dump_json(), encoding="utf-8")

    return result


# --- Async, for scoring many jobs at once ------------------------------------
# Step 4 scores 40 jobs. Done one at a time that is forty round-trips in a row;
# done concurrently it is roughly one round-trip of wall-clock. The async
# functions below mirror create()/parse() exactly, including the offline cache.

_aclient: anthropic.AsyncAnthropic | None = None


def aclient() -> anthropic.AsyncAnthropic:
    """The shared async client. Same credential resolution as client()."""
    global _aclient
    if _aclient is None:
        _aclient = anthropic.AsyncAnthropic()
    return _aclient


async def aparse(output_format, *, messages, system=None, model=None, max_tokens=None, label=None, **extra):
    """Async `parse()`. Identical semantics, including RECORD and OFFLINE.

    Note the cache key is computed the same way as the sync version, so a
    response recorded by one is replayed by the other.
    """
    model = model or MODEL
    max_tokens = max_tokens or MAX_TOKENS

    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        payload["system"] = system
    payload.update(extra)

    key = _cache_key(schema=output_format.__name__, **payload)
    cache_file = CACHE_DIR / f"parse-{key}.json"

    if OFFLINE:
        if not cache_file.exists():
            raise RuntimeError(
                f"OFFLINE=1 but no recorded response for this request (parse-{key}).\n"
                "Re-run the recorder with a live key:  RECORD=1 python record_cache.py"
            )
        return output_format.model_validate_json(cache_file.read_text("utf-8"))

    if RECORD and RESUME and cache_file.exists():
        return output_format.model_validate_json(cache_file.read_text("utf-8"))

    response = await aclient().messages.parse(output_format=output_format, **payload)
    result = response.parsed_output
    costs.record(model, response.usage, label=label or f"parse:{output_format.__name__}")

    if RECORD:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(result.model_dump_json(), encoding="utf-8")

    return result


async def aparse_full(output_format, *, messages, system=None, model=None, max_tokens=None, label=None, **extra):
    """Same as aparse(), but returns (parsed, usage) instead of just the parsed object.

    Step 4 needs the usage object to prove prompt caching is working —
    `usage.cache_read_input_tokens` is the number that makes the point. Offline
    replay has no real usage, so it reports None and the caller says so.
    """
    model = model or MODEL
    max_tokens = max_tokens or MAX_TOKENS

    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        payload["system"] = system
    payload.update(extra)

    if OFFLINE:
        return await aparse(output_format, **payload), None

    response = await aclient().messages.parse(output_format=output_format, **payload)
    costs.record(model, response.usage, label=label or f"parse:{output_format.__name__}")

    if RECORD:
        key = _cache_key(schema=output_format.__name__, **payload)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"parse-{key}.json").write_text(
            response.parsed_output.model_dump_json(), encoding="utf-8"
        )

    return response.parsed_output, response.usage
