"""The finished app — steps 5 and 6, running on any of four engines.

    streamlit run app.py

Pick the engine in the sidebar:

    Claude          the real thing, needs an API key
    Offline         recorded Claude responses, free, scripted path only
    Local model     Ollama on your own machine, free, slow
    No model        regex and keyword counting, free, instant, weak

Everything below this docstring is engine-agnostic. That is the point worth
making in the room: the model sits behind an interface (see engines/base.py),
so swapping a frontier model for keyword matching changes zero lines of UI code.
Run the same CV through Claude and through "No model", put the two match reports
next to each other, and the difference is the argument for using a model at all.

Two ideas carried over from before:

**The agent's trace is part of the product.** The sidebar shows every search and
every posting opened, live. A user who sees that trusts the result.

**The approval gate is real on every engine.** The engine can draft. It cannot
save, and it cannot open the application page. Those belong to the person whose
name is on the CV.
"""

from __future__ import annotations

import streamlit as st

import apply as apply_module
import config
import engines as engine_registry
import jobs as job_data
import tools as tool_module

st.set_page_config(page_title="Job Agent", page_icon="💼", layout="wide")

for key, default in [
    ("profile", None), ("answer", None), ("ranked", None),
    ("trace", []), ("drafts", {}), ("saved", {}), ("engine_used", None),
]:
    st.session_state.setdefault(key, default)


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("💼 Job Agent")

    if config.PUBLIC:
        # Hosted copy: only the engine that needs nothing. No key prompt, no
        # outbound calls, so an uploaded CV never leaves this server.
        engine_name = "heuristic"
        st.info("Public demo — keyword matching only, no AI model. Your CV is read "
                "in memory and never sent anywhere.", icon="🔒")
    else:
        names = [e.name for e in engine_registry.all_engines()]
        default_name = engine_registry.default_engine_name()
        locked = not config.UNLOCK_ENGINES

        engine_name = st.radio(
            "Engine",
            names,
            index=names.index(default_name) if default_name in names else 0,
            format_func=lambda n: engine_registry.ENGINES[n].label
                                  + ("" if n == default_name or not locked else "  🔒"),
            disabled=locked,
            help="What does the thinking. Everything else in the app is identical.",
        )
        # A disabled radio returns whatever sits at `index`, but be explicit
        # rather than relying on that.
        if locked:
            engine_name = default_name
            st.caption("Other engines are shown but locked for now. "
                       "`UNLOCK_ENGINES=1` to enable the picker, or `ENGINE=claude` "
                       "to run one directly.")

    engine = engine_registry.get_engine(engine_name)
    usable, why_not = engine.available()

    st.caption(engine.blurb)
    if not usable:
        st.error(why_not, icon="🚫")
    elif engine.costs_money:
        st.caption(f"Model: `{config.MODEL}` · this costs money per run.")
    if engine.name == "local":
        st.warning("Slow: roughly **4 minutes** to read the CV, **7** to search, and "
                   "**90 seconds per job** scored. Keep the scoring cap low.", icon="🐢")

    st.divider()

    sample_cvs = sorted(config.SAMPLE_CV_DIR.glob("*.pdf")) if config.SAMPLE_CV_DIR.exists() else []
    source = st.radio("Who are we searching for?",
                      ["Sample CV", "Upload my own", "Just describe it"],
                      help="No CV to hand? Describe the person in a sentence, or paste a few "
                           "skills. Every engine reads free text the same way it reads a CV.")

    cv = None
    if source == "Sample CV" and sample_cvs:
        chosen = st.selectbox("Which one", sample_cvs,
                              format_func=lambda p: p.stem.replace("_", " ").title())
        cv = engine_registry.CVSource.from_path(chosen)
    elif source == "Upload my own":
        upload = st.file_uploader("CV (PDF, TXT or MD)", type=["pdf", "txt", "md"])
        if upload:
            cv = engine_registry.CVSource(raw=upload.getvalue(), filename=upload.name)
    elif source == "Just describe it":
        typed = st.text_area(
            "Skills, a role, or a sentence about yourself",
            placeholder="Senior backend engineer, 8 years, Go and Kubernetes, "
                        "payments background, based in Herzliya.",
            height=110,
        )
        if typed.strip():
            # Free text IS a CV as far as the engines are concerned — they all
            # take a text block. Padding a bare keyword list with a heading gives
            # the profile extractor something to anchor on.
            body = typed.strip()
            if len(body) < 120 and "\n" not in body:
                body = f"Candidate\n\nSkills and experience: {body}"
            cv = engine_registry.CVSource(raw=body.encode("utf-8"), filename="typed.txt")

    # Offer the cities that actually have openings, read from the data rather
    # than hard-coded — the board set changes and a stale list sends the agent
    # looking somewhere with nothing in it.
    @st.cache_data(show_spinner=False)
    def _places() -> list[str]:
        israeli = [j for j in job_data.get_israel_jobs()
                   if job_data.looks_israeli(j["location"])]
        return job_data.cities_present(israeli)

    try:
        cities = _places()
    except Exception:  # noqa: BLE001 - no snapshot yet
        cities = []

    search_mode = st.multiselect(
        "Where to look", ["Israel — anywhere", "Remote"] + cities,
        default=["Israel — anywhere", "Remote"],
        help="Pick whole regions, specific cities, or a mix. Cities are read from "
             "the current job data, so the list changes as the boards do.",
    )

    default_cap = 6 if engine.name == "local" else 25
    score_cap = st.slider("Roles to score", 3, 40, default_cap,
                          help="Each one is a full posting read. On a local model this is the "
                               "difference between one minute and fifteen.")
    top_n = st.slider("Roles to show", 3, 15, 8)

    go = st.button("Find me jobs", type="primary", use_container_width=True,
                   disabled=cv is None or not usable)

    st.divider()
    st.subheader("Agent trace")
    trace_box = st.container()


def render_trace() -> None:
    trace_box.empty()
    with trace_box:
        for kind, text in st.session_state.trace:
            if kind == "text":
                st.markdown(f"<small>{text}</small>", unsafe_allow_html=True)
            elif kind == "tool_call":
                st.markdown(f"**→ {text}**")
            elif kind == "tool_result":
                st.caption(f"← {text}")
            elif kind == "error":
                st.error(text, icon="⚠️")


# --- The run -----------------------------------------------------------------

if go and cv is not None:
    st.session_state.update(trace=[], answer=None, ranked=None, drafts={}, saved={},
                            engine_used=engine.label)
    tool_module.JOB_REGISTRY.clear()

    def on_event(kind: str, payload: dict) -> None:
        if kind == "text":
            st.session_state.trace.append(("text", payload["text"][:400]))
        elif kind == "tool_call":
            args = ", ".join(f"{k}={v!r}" for k, v in payload["input"].items())
            st.session_state.trace.append(("tool_call", f"{payload['name']}({args})"))
        elif kind == "tool_result":
            if payload["error"]:
                st.session_state.trace.append(("error", payload["output"][:200]))
            else:
                st.session_state.trace.append(("tool_result", f"{len(payload['output'])} chars back"))
        else:
            return
        render_trace()

    try:
        with st.spinner("Reading the CV..."):
            st.session_state.profile = engine.extract_profile(cv)
        profile = st.session_state.profile

        with st.spinner("Searching..."):
            result = engine.run_agent(profile, on_event=on_event,
                                      search_modes=tuple(search_mode))
            st.session_state.answer = result["answer"]
    except Exception as exc:  # noqa: BLE001
        st.session_state.trace.append(("error", f"{type(exc).__name__}: {exc}"))
        render_trace()
        st.error(f"{engine.label} stopped: {exc}")
        st.stop()

    seen = list(tool_module.JOB_REGISTRY.values())[:score_cap]
    if seen:
        progress = st.progress(0.0, text=f"Scoring {len(seen)} roles...")
        try:
            st.session_state.ranked = engine.rank(
                profile, seen, top_n=top_n,
                on_progress=lambda d, t: progress.progress(d / t, text=f"Scored {d}/{t}"),
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not score the results: {exc}")
        progress.empty()

render_trace()


# --- Output ------------------------------------------------------------------

profile = st.session_state.profile

if profile is None:
    st.title("Find the jobs worth your time")
    st.markdown(
        "Pick a CV in the sidebar and press **Find me jobs**.\n\n"
        "An agent reads the CV, searches Israeli company boards and remote listings, opens "
        "the postings that look promising, and tells you honestly which ones fit — including "
        "what you are missing. Watch it work in the sidebar.\n\n"
        "**Try the same CV on two different engines.** The difference between *Claude* and "
        "*No model* is the whole argument for using a model."
    )
    st.stop()

st.title(profile.name)
st.caption(profile.headline)
if st.session_state.engine_used:
    st.caption(f"Engine: **{st.session_state.engine_used}**")

cols = st.columns(4)
cols[0].metric("Experience", f"{profile.years_experience:g} yrs")
cols[1].metric("Seniority", profile.seniority.title())
cols[2].metric("Remote", "Open" if profile.open_to_remote else "No")
cols[3].metric("Roles found", len(st.session_state.ranked or []))

with st.expander("What the engine understood from the CV"):
    left, right = st.columns(2)
    left.markdown("**Core skills**\n\n" + "\n".join(f"- {s}" for s in profile.core_skills))
    left.markdown("**Domains**: " + (", ".join(profile.domains) or "—"))
    right.markdown("**Target roles**\n\n" + "\n".join(f"- {r}" for r in profile.target_roles))
    right.markdown("**Can work from**: " + (", ".join(profile.locations) or "—"))
    right.markdown("**Search keywords**: `" + profile.search_keywords + "`")

if st.session_state.answer:
    with st.expander("The engine's own summary", expanded=False):
        st.markdown(st.session_state.answer)

st.divider()

ranked = st.session_state.ranked or []
if not ranked:
    st.info("No ranked roles yet.")
    st.stop()

st.subheader(f"{len(ranked)} roles, best first")

for job, match in ranked:
    with st.container(border=True):
        head, badge = st.columns([5, 1])
        head.markdown(f"### {job['title']}")
        head.markdown(f"**{job['company']}** · {job['location']}"
                      + (f" · {job['salary']}" if job["salary"] else ""))
        colour = "green" if match.score >= 75 else "orange" if match.score >= 55 else "red"
        badge.markdown(f"<h2 style='color:{colour};text-align:right'>{match.score}</h2>",
                       unsafe_allow_html=True)

        st.markdown(f"*{match.verdict}*")

        fit, gap = st.columns(2)
        fit.markdown("**Why you fit**\n\n" + "\n".join(f"- {w}" for w in match.why_you_fit))
        gap.markdown("**What you are missing**\n\n" + "\n".join(f"- {g}" for g in match.gaps))
        if match.red_flags:
            st.warning("\n".join(f"- {f}" for f in match.red_flags), icon="⚠️")

        actions = st.columns([1, 1, 3])
        actions[0].link_button("View posting", job["url"], use_container_width=True)

        job_id = job["id"]
        if actions[1].button("Draft application", key=f"draft-{job_id}", use_container_width=True):
            with st.spinner("Writing..."):
                st.session_state.drafts[job_id] = engine.draft(profile, job, match)

        if job_id in st.session_state.drafts:
            drafted = st.session_state.drafts[job_id]
            draft = apply_module.ApplicationDraft.model_validate(drafted["draft"])

            with st.expander("Draft — not saved, not sent", expanded=True):
                st.caption(f"Status: `{drafted['status']}`")
                st.markdown(f"**Subject:** {draft.subject_line}")
                st.text_area("Cover letter — select all and copy", draft.cover_letter,
                             height=200, key=f"letter-{job_id}")
                st.markdown("**Tailored CV bullets**\n\n"
                            + "\n".join(f"- {b}" for b in draft.tailored_bullets))
                st.markdown("**Talking points**\n\n"
                            + "\n".join(f"- {p}" for p in draft.talking_points))
                st.info(draft.honest_note, icon="💬")

                if config.PUBLIC:
                    # A hosted copy writes to the SERVER's disk, not the visitor's,
                    # and on a tunnelled laptop that server is your laptop. Saving
                    # is a local convenience, so it is off in public mode; the
                    # letter is right there to copy.
                    st.caption("Copy the text above — saving is disabled on the public demo.")
                elif job_id in st.session_state.saved:
                    st.success(f"Saved to `{st.session_state.saved[job_id]}` — "
                               f"finish on the company's page.")
                else:
                    # One click per job. It saves the package, logs it, and opens the
                    # company's own application form. It does NOT submit: that button
                    # is on their page and a person presses it. There is no code path
                    # here that posts an application anywhere.
                    earlier = apply_module.already_applied(job_id)
                    if earlier:
                        st.warning(f"You already opened this one on {earlier['at'][:10]} — "
                                   f"applying twice is worse than not applying.", icon="↩️")

                    save_col, apply_col = st.columns([1, 2])
                    if save_col.button("Save only", key=f"save-{job_id}",
                                       use_container_width=True):
                        folder = apply_module.save_application(profile, job, draft)
                        apply_module.record_application(profile, job, folder, opened=False)
                        st.session_state.saved[job_id] = str(folder)
                        st.rerun()

                    if apply_col.button("✅ Approve & open application page",
                                        key=f"apply-{job_id}", type="primary",
                                        use_container_width=True):
                        folder = apply_module.save_application(profile, job, draft)
                        opened = apply_module.open_application(job)
                        apply_module.record_application(profile, job, folder, opened=opened)
                        st.session_state.saved[job_id] = str(folder)
                        st.rerun()

                    st.caption("Opens the company's own form in your browser. "
                               "You paste, check, and press submit there — nothing "
                               "is sent from this app.")
