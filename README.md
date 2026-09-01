# Job Agent — live demo

An AI agent that reads a CV, searches real job listings, judges each role
honestly against the candidate, and drafts an application a human approves.

**[Live demo →](https://your-app.streamlit.app)**

```bash
pip install -r requirements.txt
streamlit run app.py
```

It runs with **no API key by default**. The keyword engine is free, instant, and
needs nothing — no account, no network calls beyond the job data itself.

## Four engines, one app

The model sits behind an interface (`engines/base.py`), so the same UI runs on
any of these. Pick one in the sidebar, or set `ENGINE=`.

| Engine | Needs | Speed | Quality |
|---|---|---|---|
| **No model** | nothing | ~1 second | Counts keywords. Cannot read. |
| **Claude** | an API key | seconds | Reads requirements properly |
| **Offline** | a recorded cache | instant | Claude, replayed |
| **Local** | Ollama + a pulled model | ~19 min | Reads and decides, weakly |

Run the same CV through two of them and compare. On the data-scientist CV:

| | No model | Claude |
|---|---|---|
| Skills found | `python go sql snowflake spark` | `llm rag pytorch python machine learning` |
| Top gap named | "No mention of Snowflake" | "No LangGraph or LangChain — named twice in the posting" |

Measured against the model's own rankings, the keyword engine reaches a Spearman
correlation of **+0.42**. Better than chance, nowhere near reading. That gap is
the argument for using a model at all — and the reason both are shipped here.

## The data

~4,900 real listings, pulled from public read-only feeds: 29 company boards
across Greenhouse, Lever and Ashby, plus remote roles from Himalayas, Jobicy and
Remotive. No scraping, no keys, no terms-of-service grey area.

```bash
python build_snapshot.py --probe   # check which boards are still alive
python build_snapshot.py           # refresh the data
```

## It drafts. It does not send.

The agent can write a tailored application. It cannot save one and it cannot
submit one — `save_application()` is simply not in the tool list, so no prompt
reaches it. Applying is a button a person presses, on the company's own page.

That is the pattern worth taking away: a guarantee comes from a capability you
never granted, not from asking a model nicely.

## Tests

```bash
pytest -q
```

26 tests, no API calls, about a second. They cover the data layer, the tool
layer, the engine interface, and a grep asserting nothing in the codebase POSTs
an application anywhere.
