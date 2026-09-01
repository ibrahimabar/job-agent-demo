"""Public job-board endpoints used by the workshop.

Everything here is a public, unauthenticated, read-only JSON endpoint. No API
keys, no scraping, no terms-of-service grey area — these are the feeds the
companies publish for exactly this purpose.

The Greenhouse tokens below were probed and confirmed live. Tokens do rot:
companies switch ATS vendors. Re-run `python build_snapshot.py --probe` the
morning of the workshop and prune anything that 404s.
"""

# Israeli tech companies whose Greenhouse board tokens are confirmed working.
# The number in the comment is the total job count at the time of probing —
# it tells you roughly what to expect, not what you'll get today.
GREENHOUSE_ISRAEL = [
    ("similarweb", "Similarweb"),      # 69
    ("riskified", "Riskified"),        # 26
    ("lightricks", "Lightricks"),      # 7
    ("jfrog", "JFrog"),                # 31
    ("gongio", "Gong"),                # 96
    ("fireblocks", "Fireblocks"),      # 69
    ("orcasecurity", "Orca Security"), # 12
    ("taboola", "Taboola"),            # 97
    ("payoneer", "Payoneer"),          # 120
    ("melio", "Melio"),                # 22
]

# Known-bad tokens, kept so nobody "helpfully" re-adds them:
# snyk, wiz, outbrain, hibob, monday, verbit, papayaglobal — all 404 on
# Greenhouse (they use a different ATS or a private board).

GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"

# --- Other applicant tracking systems ----------------------------------------
# Greenhouse is common but far from universal. Probing four providers across a
# list of candidate employers turned up boards worth thousands of roles that a
# Greenhouse-only fetcher never sees — and, deliberately, a lot of work that is
# not engineering: warehouse and delivery operations, health insurance, care
# roles, logistics, customer support, retail.
#
# Each entry is (token, display name). Probe before a workshop; tokens rot the
# same way Greenhouse ones do.

LEVER_URL = "https://api.lever.co/v0/postings/{token}?mode=json"
LEVER_BOARDS = [
    ("gopuff", "GoPuff"),          # ~760 — delivery, warehouse, retail ops
    ("walkme", "WalkMe"),          # ~35
    ("logz", "Logz.io"),           # ~2
]

ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}"
ASHBY_BOARDS = [
    ("deliveroo", "Deliveroo"),    # ~190 — operations, support, logistics
    ("hopper", "Hopper"),          # ~43
    ("lemonade", "Lemonade"),      # ~32 — insurance, incl. claims and care
    ("honeybook", "HoneyBook"),    # ~13
]

# Greenhouse boards beyond the original Israeli ten. Several are deliberately
# non-technology employers, so the results are not all engineering.
GREENHOUSE_EXTRA = [
    ("oscar", "Oscar Health"),     # ~263 — health insurance, nurses, care ops
    ("wolt", "Wolt"),              # ~248 — delivery operations, support
    ("nice", "NICE"),              # ~190
    ("vast", "Vast"),              # ~177
    ("flexport", "Flexport"),      # ~172 — logistics, freight, customs
    ("via", "Via"),                # ~159 — transit operations
    ("instacart", "Instacart"),    # ~117 — retail, shoppers, ops
    ("cabify", "Cabify"),          # ~67 — mobility operations
    ("freenow", "FREENOW"),        # ~54
    ("appsflyer", "AppsFlyer"),    # ~47
    ("forter", "Forter"),          # ~38
    ("bringg", "Bringg"),          # ~4
]

# --- Remote boards -----------------------------------------------------------
# Himalayas has by far the most listings, and almost none of them are usable
# from Israel: measured on a 1,500-job pull, 773 were United States only and
# just 1.9% were open worldwide. That is not a flaw in the feed — most remote
# work really is country-restricted — but it means volume alone does not help a
# candidate here.
#
# So we add two smaller feeds that carry a much higher share of genuinely
# worldwide roles. Measured on the same day:
#
#     Himalayas   1,500 jobs    1.9% open to an Israel-based candidate
#     Jobicy        200 jobs   16.0%
#     Remotive       19 jobs   78.9%
#
# The Himalayas listings stay regardless: a board where most "remote" roles
# quietly exclude you is exactly what the workshop's location check exists to
# catch, and it is a better teaching example than a clean feed would be.

HIMALAYAS_URL = "https://himalayas.app/jobs/api"

# Jobicy: ~200 per call, `jobGeo` says who may apply.
JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs?count={count}"

# Remotive: small but high quality; `candidate_required_location` is explicit.
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"

# Substrings that mark a location string as "in Israel". Deliberately naive —
# Step 4 of the workshop exists to show you why this list can never be complete.
ISRAEL_HINTS = [
    "israel", "tel aviv", "tel-aviv", "herzliya", "herzeliya", "haifa",
    "jerusalem", "ra'anana", "raanana", "netanya", "petah tikva", "givatayim",
    "giv'atayim", "ramat gan", "beer sheva", "be'er sheva", "yokneam",
    "rehovot", "caesarea", "hod hasharon", "kfar saba",
]


# Everything Greenhouse, in one list for the fetcher.
GREENHOUSE_BOARDS = GREENHOUSE_ISRAEL + GREENHOUSE_EXTRA
