"""What did that cost?

Every live call routed through llm.py reports its usage here, so at any point
you can ask what you have spent. Two reasons this exists:

1. Practical. The recorder makes ~120 calls and an instructor funding a room
   deserves a number, not a surprise on the invoice.
2. Pedagogical. "Cost and token budgets" is on the list of things the workshop
   skips for time, and the cheapest way to teach it is a live counter that goes
   up while students watch. Print `costs.report()` at the end of any step.

Prices are per million tokens, from the published rates. They are checked into
the repo, which means they can go stale — treat the number as a good estimate,
not an invoice.
"""

from __future__ import annotations

from datetime import date

# input, output — dollars per 1M tokens
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Claude Sonnet 5 launched with introductory pricing. Worth knowing about if you
# are choosing a model to record with while it lasts.
SONNET_5_INTRO = (2.00, 10.00)
SONNET_5_INTRO_ENDS = date(2026, 8, 31)

# Cache tokens are priced off the input rate: writing costs a premium, reading
# is the whole point of doing it.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def prices_for(model: str) -> tuple[float, float]:
    """(input, output) dollars per 1M tokens for a model id."""
    if model.startswith("claude-sonnet-5") and date.today() <= SONNET_5_INTRO_ENDS:
        return SONNET_5_INTRO
    for known, price in PRICES.items():
        if model.startswith(known):
            return price
    return (5.00, 25.00)  # unknown model: assume Opus-tier rather than under-report


class Ledger:
    """Running total of tokens and dollars, broken down by label."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, model: str, usage, label: str = "") -> float:
        """Add one API response's usage. Returns the cost of that single call."""
        if usage is None:
            return 0.0

        in_rate, out_rate = prices_for(model)
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        c_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        c_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        cost = (
            inp * in_rate
            + out * out_rate
            + c_write * in_rate * CACHE_WRITE_MULTIPLIER
            + c_read * in_rate * CACHE_READ_MULTIPLIER
        ) / 1_000_000

        self.rows.append({
            "label": label or "call", "model": model, "input": inp, "output": out,
            "cache_write": c_write, "cache_read": c_read, "cost": cost,
        })
        return cost

    # --- reading it back -----------------------------------------------------

    @property
    def total(self) -> float:
        return sum(r["cost"] for r in self.rows)

    @property
    def calls(self) -> int:
        return len(self.rows)

    def totals(self) -> dict:
        return {
            "calls": self.calls,
            "input": sum(r["input"] for r in self.rows),
            "output": sum(r["output"] for r in self.rows),
            "cache_write": sum(r["cache_write"] for r in self.rows),
            "cache_read": sum(r["cache_read"] for r in self.rows),
            "cost": self.total,
        }

    def by_label(self) -> dict[str, dict]:
        grouped: dict[str, dict] = {}
        for r in self.rows:
            g = grouped.setdefault(r["label"], {"calls": 0, "input": 0, "output": 0,
                                                "cache_read": 0, "cost": 0.0})
            g["calls"] += 1
            g["input"] += r["input"]
            g["output"] += r["output"]
            g["cache_read"] += r["cache_read"]
            g["cost"] += r["cost"]
        return grouped

    def report(self, title: str = "Cost") -> str:
        if not self.rows:
            return f"{title}: nothing billed (no live calls — cached or offline)."

        t = self.totals()
        lines = [
            f"{title}",
            "=" * 68,
            f"  {'what':<26} {'calls':>5} {'in':>9} {'out':>8} {'cached':>9} {'$':>8}",
            "  " + "-" * 66,
        ]
        for label, g in sorted(self.by_label().items(), key=lambda kv: -kv[1]["cost"]):
            lines.append(
                f"  {label[:26]:<26} {g['calls']:>5} {g['input']:>9,} {g['output']:>8,} "
                f"{g['cache_read']:>9,} {g['cost']:>8.3f}"
            )
        lines.append("  " + "-" * 66)
        lines.append(
            f"  {'TOTAL':<26} {t['calls']:>5} {t['input']:>9,} {t['output']:>8,} "
            f"{t['cache_read']:>9,} {t['cost']:>8.3f}"
        )

        if t["cache_read"]:
            in_rate, _ = prices_for(self.rows[0]["model"])
            saved = t["cache_read"] * in_rate * (1 - CACHE_READ_MULTIPLIER) / 1_000_000
            lines.append(f"\n  Prompt caching saved roughly ${saved:.3f} of that.")
        return "\n".join(lines)

    def reset(self) -> None:
        self.rows.clear()


# The shared ledger. llm.py writes to it; everything else reads it.
LEDGER = Ledger()


def record(model: str, usage, label: str = "") -> float:
    return LEDGER.record(model, usage, label)


def report(title: str = "Cost") -> str:
    return LEDGER.report(title)


def total() -> float:
    return LEDGER.total


def reset() -> None:
    LEDGER.reset()
