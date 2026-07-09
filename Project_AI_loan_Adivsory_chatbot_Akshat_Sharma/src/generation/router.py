"""
Lightweight intent router.

Decides whether a query needs the deterministic EMI calculator tool (and
extracts principal/rate/tenure if so), or is a plain policy lookup that
should go straight to retrieval + generation.

This is intentionally simple regex-based extraction rather than an LLM call,
so it works without an API key and is easy to unit test. In production you
could replace this with an LLM function-calling step for more robust
parameter extraction from messier phrasing.
"""

import re
from dataclasses import dataclass
from typing import Optional

EMI_KEYWORDS = ["emi", "monthly installment", "monthly payment", "installment amount"]

AMOUNT_RE = re.compile(r"(?:inr|rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|k)?", re.IGNORECASE)
RATE_RE = re.compile(r"([\d.]+)\s*%")
TENURE_MONTHS_RE = re.compile(r"([\d]+)\s*month", re.IGNORECASE)
TENURE_YEARS_RE = re.compile(r"([\d]+)\s*year", re.IGNORECASE)


@dataclass
class EMIParams:
    principal: Optional[float] = None
    annual_rate_pct: Optional[float] = None
    tenure_months: Optional[int] = None

    def is_complete(self) -> bool:
        return all(v is not None for v in (self.principal, self.annual_rate_pct, self.tenure_months))

    def missing_fields(self):
        missing = []
        if self.principal is None:
            missing.append("loan amount")
        if self.annual_rate_pct is None:
            missing.append("interest rate")
        if self.tenure_months is None:
            missing.append("tenure")
        return missing


def is_emi_calculation_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in EMI_KEYWORDS) or ("calculate" in q and ("loan" in q or "emi" in q))


def _parse_amount(raw_number: str, unit: str) -> float:
    val = float(raw_number.replace(",", ""))
    unit = (unit or "").lower()
    if unit in ("lakh", "lakhs"):
        val *= 100_000
    elif unit == "crore":
        val *= 10_000_000
    elif unit == "k":
        val *= 1_000
    return val


def extract_emi_params(query: str) -> EMIParams:
    params = EMIParams()

    rate_match = RATE_RE.search(query)
    if rate_match:
        params.annual_rate_pct = float(rate_match.group(1))

    months_match = TENURE_MONTHS_RE.search(query)
    years_match = TENURE_YEARS_RE.search(query)
    if months_match:
        params.tenure_months = int(months_match.group(1))
    elif years_match:
        params.tenure_months = int(years_match.group(1)) * 12

    # amount: find candidates that aren't the rate or tenure numbers already consumed
    for m in AMOUNT_RE.finditer(query):
        raw, unit = m.group(1), m.group(2)
        if not raw or raw.strip(",.") == "":
            continue
        # look only at the very next word (not the whole rest of the query) to decide
        # if this number is actually the rate (e.g. "10.5%") or tenure (e.g. "3 years")
        after = query[m.end():m.end() + 8]
        if after.lstrip().startswith("%"):
            continue
        next_word = after.strip().split(" ")[0] if after.strip() else ""
        if next_word.lower().startswith("month") or next_word.lower().startswith("year"):
            continue
        val = _parse_amount(raw, unit)
        if val >= 1000:  # loan amounts are realistically >= 1000
            params.principal = val
            break

    return params


if __name__ == "__main__":
    tests = [
        "What is the EMI for a loan of 5 lakh at 11.75% for 48 months?",
        "Calculate my monthly installment for a 20,00,000 loan over 5 years at 9.1%",
        "What is the minimum CIBIL score for a personal loan?",  # not an EMI query
        "Calculate EMI for 750000 at 10.5% for 3 years",
    ]
    for t in tests:
        is_emi = is_emi_calculation_query(t)
        print(f"\nQuery: {t}")
        print(f"  Is EMI query: {is_emi}")
        if is_emi:
            params = extract_emi_params(t)
            print(f"  Extracted: principal={params.principal}, rate={params.annual_rate_pct}, "
                  f"tenure_months={params.tenure_months}, complete={params.is_complete()}")
