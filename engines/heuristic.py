"""The engine with no model in it at all. Free, instant, and visibly worse.

Everything here is regex, keyword vocabularies and arithmetic. No API, no key,
no network, no GPU. It is the honest answer to "could you just do this with
string matching?" — yes, and here is exactly how far that gets you.

Keep this file open next to `claude.py` during the workshop. The interesting
comparison is not that this one is worse; it is *where* it is worse:

    reading a CV        heuristic does fine on a clean one-column PDF,
                        and falls apart on anything designed
    extracting skills   good, as long as the skill is in the vocabulary below.
                        It cannot recognise a technology invented last year.
    deciding to search  it cannot. The searches here are hard-coded by a human.
    judging a match     it counts words. It cannot tell a hard requirement from
                        a nice-to-have, and it never knows *why* something fits.
    writing a letter    it fills in a template. Everyone gets the same letter.

That last pair is where the money is. A keyword counter can tell you a posting
mentions Go. Only a reader can tell you the posting needs three years of Rust
you do not have.
"""

from __future__ import annotations

import re
from collections import Counter

import jobs as job_data
import tools as tool_module
from apply import ApplicationDraft
from profile import CandidateProfile
from scoring import JobMatch

from .base import CVSource, Engine, split_search_modes

# --- Vocabularies ------------------------------------------------------------
# This list IS the engine's world. Anything not in it does not exist. That is
# the fundamental limitation of the approach, and no amount of extra entries
# fixes it — you are always one new framework behind.

SKILLS = [
    "Python", "Go", "Golang", "Java", "JavaScript", "TypeScript", "Ruby", "Rust", "C++", "C#",
    "PHP", "Scala", "Kotlin", "Swift", "SQL", "Bash", "R", "MATLAB",
    "React", "Vue", "Angular", "Next.js", "Node.js", "Django", "Flask", "FastAPI", "Rails",
    "Spring", "Express", "GraphQL", "REST", "gRPC",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "ClickHouse", "Cassandra",
    "DynamoDB", "Snowflake", "BigQuery", "Vitess", "Kafka", "RabbitMQ", "Flink", "Spark",
    "Airflow", "dbt", "Databricks",
    "Kubernetes", "Docker", "Terraform", "Ansible", "ArgoCD", "Jenkins", "GitLab", "GitHub Actions",
    "AWS", "GCP", "Azure", "Prometheus", "Grafana", "Datadog", "Linux",
    "PyTorch", "TensorFlow", "scikit-learn", "Hugging Face", "Transformers", "XGBoost",
    "LangChain", "RAG", "LLM", "NLP", "pandas", "NumPy", "SageMaker", "Pinecone", "pgvector",
    "HTML", "CSS", "Tailwind", "SASS", "Webpack", "Vite", "Jest", "Cypress", "Playwright",
    "Figma", "Storybook", "Redux", "accessibility", "WCAG",
]

SENIORITY_WORDS = [
    ("principal", "principal"), ("staff", "staff"), ("director", "director"),
    ("head of", "director"), ("vp ", "director"), ("lead", "senior"),
    ("senior", "senior"), ("sr.", "senior"), ("mid-level", "mid"), ("junior", "junior"),
    ("jr.", "junior"), ("intern", "intern"), ("graduate", "junior"), ("entry", "junior"),
]

DOMAIN_HINTS = {
    "fintech": ["payment", "fintech", "banking", "ledger", "transaction", "billing", "fraud"],
    "cybersecurity": ["security", "threat", "vulnerability", "siem", "soc ", "malware"],
    "healthcare": ["clinical", "healthcare", "medical", "patient", "health"],
    "adtech": ["advertis", "adtech", "campaign", "programmatic"],
    "e-commerce": ["e-commerce", "ecommerce", "checkout", "storefront", "retail"],
    "devtools": ["developer tools", "ci/cd", "platform engineering", "sdk"],
    "data": ["data platform", "analytics", "warehouse", "etl", "pipeline"],
}

LANGUAGES = ["English", "Hebrew", "Arabic", "Russian", "French", "Spanish", "German"]

WORLDWIDE_HINTS = ("worldwide", "israel", "emea", "middle east", "europe", "anywhere")

# Different spellings of the same skill. Without this, a CV saying "Golang" and
# a posting saying "Go" look like no overlap at all.
ALIASES = {
    "Go": ["golang"], "Kubernetes": ["k8s"], "JavaScript": ["js", "ecmascript"],
    "PostgreSQL": ["postgres", "psql"], "TypeScript": ["ts"], "Node.js": ["nodejs", "node"],
    "scikit-learn": ["sklearn"], "Hugging Face": ["huggingface"], "LLM": ["llms", "large language model"],
    "NLP": ["natural language processing"], "AWS": ["amazon web services"],
    "GCP": ["google cloud"], "CI/CD": ["cicd"],
}

# Short tokens that are also ordinary English. "go to production", "a c program",
# "the r in RAG" — matching these lowercase produces confident nonsense, so they
# only count when they appear capitalised the way a technology is written.
CASE_SENSITIVE = {"Go", "R", "C", "C++", "C#", "Rust", "Spark", "Flink", "Redis"}


def _has(term: str, text: str) -> bool:
    """Word-boundary match. `"go" in text` is true for Google, Mongo and algorithm."""
    return re.search(rf"\b{re.escape(term.lower())}\b", text) is not None


class HeuristicEngine(Engine):
    name = "heuristic"
    label = "No model"
    blurb = "Keyword matching. Free, instant, and no AI involved."
    costs_money = False

    def available(self) -> tuple[bool, str]:
        return True, ""

    # --- Reading the CV ------------------------------------------------------

    def _mentions(self, skill: str, text: str) -> int:
        """How many times this skill appears, counting its aliases.

        Case-sensitive for the ambiguous short ones — otherwise "go to
        production" makes a data scientist look like a Go developer, which is
        exactly what the first version of this file did.
        """
        forms = [skill] + ALIASES.get(skill, [])
        total = 0
        for form in forms:
            if skill in CASE_SENSITIVE and form == skill:
                total += len(re.findall(rf"\b{re.escape(form)}\b", text))
            else:
                total += len(re.findall(rf"\b{re.escape(form.lower())}\b", text.lower()))
        return total

    def _skills_in(self, text: str) -> list[str]:
        """Skills this CV shows evidence for, strongest first.

        The ordering is the whole point. Version one returned them in the order
        they happened to sit in the SKILLS list, then truncated to 15 — so a
        machine-learning CV came back as "python, go, sql, snowflake, spark"
        with PyTorch, RAG and LLM cut off the end. Rank by evidence instead:
        how often the skill appears, with a bonus for appearing in a section
        that looks like a skills list.
        """
        skills_section = ""
        match = re.search(r"(?:^|\n)\s*(?:technical\s+)?skills?\b(.{0,600})",
                          text, re.I | re.S)
        if match:
            skills_section = match.group(1)

        scored = []
        for skill in SKILLS:
            count = self._mentions(skill, text)
            if not count:
                continue
            weight = count + (2 if self._mentions(skill, skills_section) else 0)
            scored.append((weight, skill))

        scored.sort(key=lambda pair: (-pair[0], SKILLS.index(pair[1])))
        found = [s for _, s in scored]
        return [s for s in found if s != "Golang"]

    def _years(self, text: str) -> float:
        """Years of experience, by three increasingly desperate methods."""
        # 1. Stated outright.
        stated = re.search(r"(\d{1,2})\+?\s*years?\s+(?:of\s+)?experience", text, re.I)
        if stated:
            return float(stated.group(1))
        # 2. Written as a word.
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
        spelled = re.search(r"\b(" + "|".join(words) + r")\s+years?\s+of\s+experience", text, re.I)
        if spelled:
            return float(words[spelled.group(1).lower()])
        # 3. Earliest four-digit year in the document, assume they started then.
        years = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-4]\d)\b", text)]
        if years:
            from datetime import date

            return max(0.0, float(date.today().year - min(years)))
        return 0.0

    def _seniority(self, text: str, years: float) -> str:
        """Read the candidate's own most recent job title — not any nearby word.

        Version one scanned the first 1200 characters for any seniority word,
        which classified a two-year junior developer as "senior" because the
        word appeared in a summary sentence about the team she worked with.
        Look at title-shaped lines only, and let years of experience overrule a
        title that disagrees with them.
        """
        title_lines = [
            l.strip() for l in text.splitlines()[:40]
            if re.search(r"\b(engineer|developer|scientist|manager|analyst|architect|lead)\b", l, re.I)
            and len(l.strip()) < 90
        ]
        for line in title_lines[:3]:
            low_line = line.lower()
            for needle, level in SENIORITY_WORDS:
                if needle in low_line:
                    # A title can lie or belong to someone else. Two years is not
                    # staff, whatever the line says — but only override when we
                    # actually measured some years. Free text like "senior backend
                    # engineer, Go, Kubernetes" states a level and no dates, and
                    # treating that absence as evidence of juniority is wrong.
                    if years > 0:
                        if level in ("staff", "principal", "director") and years < 6:
                            break
                        if level == "senior" and years < 4:
                            break
                    return level
        if years >= 10:
            return "staff"
        if years >= 5:
            return "senior"
        if years >= 2:
            return "mid"
        return "junior"

    def extract_profile(self, cv: CVSource) -> CandidateProfile:
        text = cv.as_text()
        low = text.lower()
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # The name is almost always the first line. "Almost" is doing real work.
        name = lines[0] if lines else "Unknown"
        if len(name) > 48 or "@" in name:
            name = "Unknown"

        skills = self._skills_in(text)
        years = self._years(text)
        seniority = self._seniority(text, years)

        domains = [d for d, hints in DOMAIN_HINTS.items() if any(h in low for h in hints)]
        languages = [l for l in LANGUAGES if _has(l, low)] or ["English"]
        locations = sorted({
            h.title() for h in job_data.ISRAEL_HINTS if h in low
        }) if any(h in low for h in job_data.ISRAEL_HINTS) else []
        if "israel" in low and "Israel" not in locations:
            locations.append("Israel")
        if "remote" in low:
            locations.append("Remote")

        # A role title, assembled rather than understood.
        area = "Backend"
        if any(_has(s, low) for s in ("React", "Vue", "Angular", "CSS")):
            area = "Frontend"
        if any(_has(s, low) for s in ("PyTorch", "TensorFlow", "scikit-learn", "LLM")):
            area = "Machine Learning"
        if any(_has(s, low) for s in ("Kubernetes", "Terraform", "ArgoCD")) and area == "Backend":
            area = "Backend/Infrastructure"

        prefix = {"staff": "Staff", "principal": "Principal", "senior": "Senior",
                  "director": "Director of", "mid": "", "junior": "Junior",
                  "intern": "Intern"}.get(seniority, "")
        target = [f"{prefix} {area} Engineer".strip(),
                  f"{area} Engineer",
                  f"Senior {area} Engineer"]

        # Anything with a number in it is probably an achievement. Crude, works.
        notable = [l for l in lines if re.search(r"\d+\s*(%|x\b|ms\b|TB|GB|/sec)", l)][:4]

        return CandidateProfile(
            name=name,
            headline=f"{prefix} {area.lower()} engineer, {years:g} years, "
                     f"{', '.join(skills[:3]) or 'no skills detected'}".strip(),
            years_experience=years,
            seniority=seniority,
            core_skills=skills[:15] or ["(none matched the vocabulary)"],
            domains=domains,
            target_roles=list(dict.fromkeys(target)),
            search_keywords=" ".join(s.lower() for s in skills[:5]) or area.lower(),
            languages=languages,
            open_to_remote="remote" in low,
            locations=list(dict.fromkeys(locations)) or ["Unknown"],
            notable=notable or ["(no quantified achievements detected)"],
        )

    def summarize_cv(self, cv: CVSource) -> str:
        """A template, filled in. Note it says nothing a human could not have
        computed with `grep -c`, and nothing about whether this person is good."""
        p = self.extract_profile(cv)
        return (
            f"**{p.name}** — {p.headline}.\n\n"
            f"Matched **{len(p.core_skills)}** known technologies: {', '.join(p.core_skills[:10])}.\n"
            f"Domains detected from keywords: {', '.join(p.domains) or 'none'}.\n"
            f"Estimated seniority: **{p.seniority}** ({p.years_experience:g} years).\n\n"
            f"_This summary was assembled by string matching. It has not read anything — "
            f"it cannot tell you whether the experience is deep or padded, whether the "
            f"career makes sense, or what this person should do next._"
        )

    # --- Searching -----------------------------------------------------------

    def run_agent(self, profile, *, on_event=None, search_modes=("Israel", "Remote")) -> dict:
        """A fixed sequence of searches. There is no agent here.

        This is the part that cannot be faked with rules and is worth saying out
        loud: a human wrote this plan, in advance, without seeing the results. It
        cannot look at what came back and try different words, which is precisely
        what the real agent does on every run.
        """
        def emit(kind, payload):
            if on_event:
                on_event(kind, payload)

        emit("text", {"text": "No model in this mode — running a fixed search plan "
                              "written by a human who has not seen the results."})

        want_israel, want_remote, cities = split_search_modes(search_modes)

        # If no city was picked but the CV names one we recognise, search it too —
        # a candidate who can only reach Herzliya is not served by Tel Aviv results.
        if want_israel and not cities:
            home = next((c for c in job_data.known_cities()
                         if any(job_data.normalize_place(c) in job_data.normalize_place(l)
                                for l in profile.locations)), "")
            if home:
                cities = [home]

        queries = []
        if want_israel:
            queries += [("search_israel_jobs", profile.search_keywords, ""),
                        ("search_israel_jobs", " ".join(profile.core_skills[:3]).lower(), "")]
            for city in cities:
                queries.append(("search_israel_jobs", profile.search_keywords, city))
        if want_remote:
            queries += [("search_remote_jobs", profile.search_keywords, ""),
                        ("search_remote_jobs", " ".join(profile.core_skills[:3]).lower(), "")]

        for tool_name, keywords, location in queries:
            args = {"keywords": keywords, "limit": 12}
            if location:
                args["location"] = location
            emit("tool_call", {"name": tool_name, "input": args})
            output, is_error = tool_module.run_tool(tool_name, args)
            emit("tool_result", {"name": tool_name, "output": output, "error": is_error})

        seen = len(tool_module.JOB_REGISTRY)
        answer = (
            f"**Keyword mode — no model was used.**\n\n"
            f"Ran {len(queries)} fixed searches and found {seen} distinct roles. They are "
            f"ranked below by how many of your detected skills appear in each posting.\n\n"
            f"What this mode cannot do: notice that its first search was bad and try "
            f"different words, read a requirements list, tell a hard requirement from a "
            f"nice-to-have, or explain *why* anything fits. Those are the jobs a model does."
        )
        emit("text", {"text": answer})
        emit("done", {"answer": answer, "turns": len(queries)})
        return {"answer": answer, "messages": [], "turns": len(queries), "tool_calls": []}

    # --- Judging -------------------------------------------------------------

    def _location_ok(self, job: dict, profile) -> bool:
        if job["source"] == "himalayas":
            restriction = job["location"].replace("Remote — ", "").lower()
            return any(w in restriction for w in WORLDWIDE_HINTS)
        return job_data.looks_israeli(job["location"])

    # Headings that mean "this is what we actually require", as opposed to the
    # paragraph about snacks. Finding this section is the closest a keyword
    # engine gets to telling a dealbreaker from a nice-to-have.
    REQ_HEADINGS = re.compile(
        r"(requirements|qualifications|what you.{0,12}(bring|need|have)|"
        r"who you are|must have|we.{0,4}re looking for|your experience|skills? (?:and|&) )",
        re.I,
    )

    def _requirements_of(self, description: str) -> str:
        """The chunk of a posting after a requirements-ish heading."""
        match = self.REQ_HEADINGS.search(description)
        if not match:
            return ""
        return description[match.start(): match.start() + 2500]

    def _years_required(self, text: str) -> float | None:
        match = re.search(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?years?", text, re.I)
        return float(match.group(1)) if match else None

    def _idf(self, jobs: list[dict]) -> dict[str, float]:
        """How rare is each skill across this pool of postings?

        Without this, matching "SQL" counts the same as matching "Vitess". One
        of those appears in half the postings on the board and tells you nothing;
        the other is a real signal. This is ordinary TF-IDF, and it is the single
        biggest improvement available to a keyword engine.
        """
        import math

        n = max(1, len(jobs))
        blobs = [f"{j['title']} {j['description'][:4000]}".lower() for j in jobs]
        idf = {}
        for skill in SKILLS:
            df = sum(1 for b in blobs if _has(skill, b))
            idf[skill] = math.log((n + 1) / (df + 1)) + 1.0
        return idf

    def score(self, profile, job: dict, idf: dict[str, float] | None = None) -> JobMatch:
        """Rank on a plain weighted count. Explain with everything else.

        This split is the result of actually measuring it. Ranked against the
        same jobs scored by Claude (Spearman rho, two CVs, 14 jobs each):

            plain count of matching skills          rho = +0.43
            + years/seniority penalties             rho = +0.35
            TF-IDF weighting, absolute match        rho = +0.16
            TF-IDF weighting, coverage ratio        rho = +0.00

        Every clever thing made the ordering worse. Coverage ratio was the worst
        because it rewards postings that ask for very little — match two of three
        listed technologies and you outrank a job that wants fifteen and matches
        eight of them.

        So the count does the ranking. IDF and the requirements section are still
        earned their place, but for *explaining*: naming the rarest matches first
        and telling the candidate what the posting asks for that they lack. The
        years and seniority checks are reported as red flags without moving the
        score, because they are useful to read and were measurably bad at sorting.

        The honest headline: rho +0.43 is a weak-to-moderate correlation. No
        amount of term weighting closed the gap, because the gap is reading.
        """
        idf = idf or {}
        title = job["title"].lower()
        description = job["description"]
        body = description.lower()
        blob = f"{title} {body}"
        requirements = self._requirements_of(description).lower()

        hits = [s for s in profile.core_skills if _has(s, blob)]
        title_hits = [s for s in profile.core_skills if _has(s, title)]

        # --- the ranking signal, and nothing else ---------------------------
        score = 24 + 5 * len(hits) + 10 * len(title_hits)

        # --- everything below only explains; it does not sort ---------------
        red_flags: list[str] = []

        needed = self._years_required(requirements or body[:1500])
        if needed and profile.years_experience and needed - profile.years_experience >= 2:
            red_flags.append(
                f"Asks for {needed:g}+ years; your CV shows {profile.years_experience:g}")

        job_level = next((lvl for word, lvl in SENIORITY_WORDS if word in title), None)
        order = ["intern", "junior", "mid", "senior", "staff", "principal", "director"]
        if job_level and profile.seniority in order and job_level in order:
            gap = order.index(job_level) - order.index(profile.seniority)
            if gap <= -2:
                red_flags.append(f"Titled {job_level} — you would read as overqualified")
            elif gap >= 2:
                red_flags.append(f"Titled {job_level} — a stretch above your detected level")

        location_ok = self._location_ok(job, profile)
        if not location_ok:
            red_flags.append(f"Location restriction: {job['location'][:70]}")

        # Rarity orders the explanation: "Vitess" is worth saying, "SQL" is not.
        def rarity(skill: str) -> float:
            return idf.get(skill, 1.0) * (3.0 if _has(skill, title) else 1.0)

        wanted = [s for s in SKILLS if _has(s, requirements)] if requirements else []
        missing = sorted([s for s in wanted if s not in profile.core_skills],
                         key=rarity, reverse=True)

        domain_overlap = [d for d in profile.domains
                          if any(h in body for h in DOMAIN_HINTS.get(d, []))]

        why = [f"{s} — on your CV and in this posting"
               for s in sorted(hits, key=rarity, reverse=True)[:4]]
        if domain_overlap:
            why.append(f"Same domain as your background: {', '.join(domain_overlap)}")

        gaps = [f"{s} — in the requirements, not on your CV" for s in missing[:3]]
        if not gaps:
            gaps = [f"No mention of {s}" for s in profile.core_skills[:3]
                    if not _has(s, blob)][:3]

        return JobMatch(
            score=max(5, min(96, score)),
            verdict=(f"{len(hits)} of your skills appear here"
                     + (f", including {', '.join(title_hits)} in the title" if title_hits else "")
                     + ". Counted, not read."),
            why_you_fit=why or ["No skill overlap detected"],
            gaps=gaps or ["(nothing detected)"],
            red_flags=red_flags,
            location_ok=location_ok,
        )

    def rank(self, profile, jobs: list[dict], *, top_n: int = 8, on_progress=None):
        idf = self._idf(jobs)          # used for explanation ordering, not scoring
        scored = []
        for i, job in enumerate(jobs, start=1):
            scored.append((job, self.score(profile, job, idf)))
            if on_progress:
                on_progress(i, len(jobs))
        scored = [(j, m) for j, m in scored if m.location_ok]
        scored.sort(key=lambda pair: -pair[1].score)
        return scored[:top_n]

    # --- Writing -------------------------------------------------------------

    def draft(self, profile, job: dict, match=None) -> dict:
        """A mail-merge. Every candidate gets the same letter with the nouns swapped."""
        overlap = [s for s in profile.core_skills
                   if _has(s, f"{job['title']} {job['description']}".lower())][:4]

        letter = (
            f"Dear {job['company']} hiring team,\n\n"
            f"I am applying for the {job['title']} role. I have {profile.years_experience:g} years "
            f"of experience as a {profile.seniority} engineer"
            + (f", working with {', '.join(overlap)}" if overlap else "")
            + ".\n\n"
            + (f"My background includes {profile.notable[0].lower()}. " if profile.notable else "")
            + f"I am based in {profile.locations[0]} and "
            + ("open to remote work" if profile.open_to_remote else "looking locally")
            + ".\n\nI would welcome the chance to discuss the role.\n\n"
            f"Best regards,\n{profile.name}"
        )

        return {
            "status": "awaiting_approval",
            "job_id": job["id"],
            "job_title": job["title"],
            "company": job["company"],
            "apply_url": job["url"],
            "draft": ApplicationDraft(
                subject_line=f"Application — {job['title']}",
                cover_letter=letter,
                tailored_bullets=(profile.notable[:3] or ["(no achievements detected in the CV)"]),
                talking_points=[f"Your experience with {s}" for s in overlap[:3]]
                               or ["(no overlap detected)"],
                honest_note=("This letter came out of a template. It says nothing specific about "
                             "this company, because keyword matching cannot read a job posting and "
                             "find the thing worth mentioning. Send it and a recruiter will know."),
            ).model_dump(),
            "note": "Template output. Nothing saved, nothing submitted — the gate still applies.",
        }
