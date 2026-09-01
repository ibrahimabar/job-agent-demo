"""Generate the three sample CVs as real PDFs.

Run once during setup: `python make_sample_cvs.py`

These exist so that nobody is blocked on having a CV handy, and — more
importantly — so nobody feels pressured to upload their own private CV through
a shared workshop API key. The people are invented; the skills, tools and
career shapes are realistic enough to produce genuinely different search
results, which is what makes the demo interesting.
"""

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from config import SAMPLE_CV_DIR

_base = getSampleStyleSheet()
NAME = ParagraphStyle("Name", parent=_base["Title"], fontSize=20, spaceAfter=2, alignment=TA_LEFT)
CONTACT = ParagraphStyle("Contact", parent=_base["Normal"], fontSize=9, textColor="#555555", spaceAfter=10)
H = ParagraphStyle("H", parent=_base["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4, textColor="#1a3d6d")
BODY = ParagraphStyle("Body", parent=_base["Normal"], fontSize=9.5, leading=13)
ROLE = ParagraphStyle("Role", parent=_base["Normal"], fontSize=10, leading=13, spaceBefore=6, spaceAfter=1)


def render(filename: str, cv: dict) -> None:
    SAMPLE_CV_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_CV_DIR / filename
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=cv["name"], author=cv["name"],
    )

    story = [Paragraph(cv["name"], NAME), Paragraph(cv["contact"], CONTACT)]

    story += [Paragraph("Summary", H), Paragraph(cv["summary"], BODY)]

    story += [Paragraph("Experience", H)]
    for job in cv["experience"]:
        story.append(Paragraph(f"<b>{job['title']}</b> — {job['company']}, {job['where']}  ·  {job['dates']}", ROLE))
        story.append(
            ListFlowable(
                [ListItem(Paragraph(b, BODY), leftIndent=10) for b in job["bullets"]],
                bulletType="bullet", start="•", leftIndent=12, bulletFontSize=7,
            )
        )

    story += [Paragraph("Skills", H), Paragraph(cv["skills"], BODY)]
    story += [Paragraph("Education", H), Paragraph(cv["education"], BODY)]
    if cv.get("extra"):
        story += [Paragraph(cv["extra_title"], H), Paragraph(cv["extra"], BODY)]
    story += [Spacer(1, 6), Paragraph("Languages: " + cv["languages"], BODY)]

    doc.build(story)
    print(f"  wrote {path.relative_to(SAMPLE_CV_DIR.parent)}")


JUNIOR_FRONTEND = {
    "name": "Maya Ben-Ari",
    "contact": "Tel Aviv, Israel · maya.benari@example.com · +972 5X-XXX-XXXX · github.com/example",
    "summary": (
        "Front-end developer with just under two years of professional experience building "
        "React interfaces for consumer products. Comfortable owning a feature from Figma "
        "handoff through to production, and increasingly drawn to the performance and "
        "accessibility side of the work. Looking for a team where code review is taken "
        "seriously and there is someone more senior to learn from."
    ),
    "experience": [
        {
            "title": "Front-End Developer", "company": "Kavana Retail", "where": "Tel Aviv",
            "dates": "Mar 2025 – present",
            "bullets": [
                "Build and maintain the customer-facing storefront in React 18 and TypeScript, "
                "serving roughly 40,000 monthly active shoppers.",
                "Cut the product listing page's Largest Contentful Paint from 4.1s to 1.6s by "
                "replacing a client-side filter with a server-rendered one and lazy-loading imagery.",
                "Introduced Playwright end-to-end tests for the checkout funnel after a pricing bug "
                "reached production; the suite has caught four regressions since.",
                "Pair regularly with the design team to keep a shared component library in Storybook.",
            ],
        },
        {
            "title": "Junior Developer (part-time)", "company": "Hadar Studio", "where": "Ramat Gan",
            "dates": "Sep 2024 – Feb 2025",
            "bullets": [
                "Delivered marketing sites for six clients using Next.js and Tailwind CSS.",
                "Wrote the studio's first accessibility checklist after an audit flagged contrast "
                "and keyboard-navigation failures across most of the portfolio.",
            ],
        },
    ],
    "skills": (
        "<b>Languages:</b> JavaScript, TypeScript, HTML, CSS, some Python<br/>"
        "<b>Frameworks:</b> React, Next.js, Tailwind CSS, Redux Toolkit, React Query<br/>"
        "<b>Testing:</b> Playwright, Vitest, React Testing Library<br/>"
        "<b>Tooling:</b> Vite, Git, Figma, Storybook, GitHub Actions, Vercel"
    ),
    "education": "B.Sc. Computer Science, The Open University of Israel — 2024. Final project: a "
                 "browser extension that audits pages for WCAG contrast violations.",
    "extra_title": "Side work",
    "extra": "Maintain a small open-source React hook library for form validation (about 300 GitHub stars). "
             "Wrote a well-shared blog post on why the studio abandoned CSS-in-JS.",
    "languages": "Hebrew (native), English (fluent), Arabic (conversational)",
}

SENIOR_BACKEND = {
    "name": "Daniel Rozen",
    "contact": "Herzliya, Israel · d.rozen@example.com · linkedin.com/in/example · Open to remote",
    "summary": (
        "Backend and infrastructure engineer with nine years of experience, the last four spent "
        "on payment and fraud systems where correctness is not negotiable. Comfortable being the "
        "person who is paged. Most interested in roles where the hard part is the distributed "
        "system rather than the feature list, and where I can keep mentoring rather than move "
        "fully into management."
    ),
    "experience": [
        {
            "title": "Staff Backend Engineer", "company": "Tovana Payments", "where": "Tel Aviv",
            "dates": "Jan 2023 – present",
            "bullets": [
                "Technical owner of the transaction ledger — Go services on Kubernetes handling "
                "roughly 2,500 payments per second at peak with a hard exactly-once requirement.",
                "Led the migration from a single PostgreSQL primary to Vitess-sharded MySQL across "
                "seven months with no customer-visible downtime and no data loss.",
                "Rebuilt the fraud-scoring pipeline on Kafka and Flink, bringing decision latency "
                "from 400ms down to 45ms at p99 and letting the risk team ship rules themselves.",
                "Introduced an error-budget policy that cut the paging rate by more than half in "
                "two quarters, mostly by deleting alerts nobody acted on.",
                "Mentor three mid-level engineers; run the internal design-review forum.",
            ],
        },
        {
            "title": "Senior Backend Engineer", "company": "Sela Security", "where": "Ra'anana",
            "dates": "Jun 2019 – Dec 2022",
            "bullets": [
                "Built the ingestion tier for a cloud security product — Python and Go, ingesting "
                "about 8TB of audit logs daily from AWS, GCP and Azure.",
                "Designed a multi-tenant rate limiter that survived a customer's 30x traffic spike "
                "without spilling over into other tenants.",
                "Took the service from manual deploys to Terraform and ArgoCD; deploy frequency "
                "went from fortnightly to several times a day.",
            ],
        },
        {
            "title": "Backend Developer", "company": "Amdocs", "where": "Ra'anana",
            "dates": "Aug 2017 – May 2019",
            "bullets": [
                "Java microservices for telecom billing; wrote the reconciliation job that finally "
                "closed a long-running revenue-leakage discrepancy.",
            ],
        },
    ],
    "skills": (
        "<b>Languages:</b> Go, Python, Java, SQL, Bash<br/>"
        "<b>Data:</b> PostgreSQL, MySQL/Vitess, Kafka, Flink, Redis, ClickHouse<br/>"
        "<b>Infrastructure:</b> Kubernetes, Terraform, ArgoCD, AWS, GCP, Prometheus, Grafana<br/>"
        "<b>Practices:</b> distributed systems design, SRE and on-call, incident review, mentoring"
    ),
    "education": "B.Sc. Software Engineering, Technion — Israel Institute of Technology, 2017.",
    "extra_title": "Talks",
    "extra": "\"Sharding a ledger without losing a cent\" — Go Israel meetup, 2025. "
             "Occasional contributor to the Vitess project.",
    "languages": "Hebrew (native), English (fluent), Russian (conversational)",
}

DATA_SCIENTIST = {
    "name": "Noa Shapira",
    "contact": "Haifa, Israel · noa.shapira@example.com · Open to remote and hybrid",
    "summary": (
        "Machine learning engineer with five years of experience, focused on taking models past "
        "the notebook and into systems people actually depend on. Half my work is modelling and "
        "half is the evaluation, monitoring and data plumbing that decides whether the modelling "
        "was worth anything. Recently spent a lot of time on LLM applications and their evaluation."
    ),
    "experience": [
        {
            "title": "Senior ML Engineer", "company": "Refua Health AI", "where": "Haifa",
            "dates": "Apr 2023 – present",
            "bullets": [
                "Own the clinical note summarization product end to end: retrieval, prompting, "
                "evaluation harness and the human review queue used by twelve partner clinics.",
                "Built an LLM-as-judge evaluation suite over 1,200 clinician-labelled examples; it "
                "caught a regression that would otherwise have shipped a 9-point drop in factuality.",
                "Reduced inference cost per note by 62% through prompt caching, aggressive "
                "retrieval pruning, and moving the easy cases to a smaller model.",
                "Work directly with clinicians on failure review — the single highest-value hour "
                "in my week and the source of most real model improvements.",
            ],
        },
        {
            "title": "Data Scientist", "company": "Migdal Analytics", "where": "Tel Aviv",
            "dates": "Feb 2021 – Mar 2023",
            "bullets": [
                "Built churn and lifetime-value models for subscription clients in PyTorch and "
                "scikit-learn; the churn model drove a retention campaign worth about $1.4M annually.",
                "Replaced a fragile collection of notebooks with an Airflow and dbt pipeline, which "
                "made the weekly client reports reproducible for the first time.",
                "Ran the A/B testing framework and, more often than anyone liked, explained why a "
                "result was not significant.",
            ],
        },
    ],
    "skills": (
        "<b>Languages:</b> Python, SQL, some Go<br/>"
        "<b>ML:</b> PyTorch, scikit-learn, Hugging Face Transformers, XGBoost<br/>"
        "<b>LLM work:</b> RAG, evaluation harnesses, prompt engineering, vector search (pgvector, Pinecone)<br/>"
        "<b>Data and platform:</b> Airflow, dbt, Spark, Snowflake, Docker, AWS SageMaker, Weights &amp; Biases"
    ),
    "education": "M.Sc. Computer Science (machine learning), Technion, 2021. Thesis on uncertainty "
                 "estimation for medical image classifiers. B.Sc. Statistics, Hebrew University, 2019.",
    "extra_title": "Publications",
    "extra": "Second author on a workshop paper about calibration under distribution shift (NeurIPS "
             "workshop, 2022). Write an occasional newsletter on LLM evaluation with ~2,000 readers.",
    "languages": "Hebrew (native), English (fluent)",
}


if __name__ == "__main__":
    print("Generating sample CVs...")
    render("junior_frontend.pdf", JUNIOR_FRONTEND)
    render("senior_backend.pdf", SENIOR_BACKEND)
    render("data_scientist.pdf", DATA_SCIENTIST)
    print("\nAll three are invented people. Use them instead of your own CV when "
          "the workshop is running on a shared API key.")
