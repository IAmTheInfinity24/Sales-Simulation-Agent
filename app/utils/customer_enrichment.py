"""
Customer Enrichment
-------------------
NEW in v2 — was completely absent in v1.

The customers.csv has 100K rows with rich signals that were completely
unused: churn_risk, credit_score, loyalty_points, avg_order_value,
favorite_category, device_type, subscription.

This module:
1. Loads the customer CSV once (lazy, cached).
2. look_up_customer() fuzzy-matches by name and returns enrichment signals.
3. build_enrichment_context() converts those signals into:
   - A score bias dict for the recommendation scorer (e.g. boost premium brands
     for high-income / high-credit customers).
   - A prose snippet for the LLM prompt.
   - A category preference hint for the Pinecone query filter.

If the customer is not found in the CSV (e.g. new customer, name mismatch),
the functions gracefully return empty/neutral values.
"""

from __future__ import annotations

import os
import pandas as pd
from functools import lru_cache

_CUSTOMER_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "customers.csv")


@lru_cache(maxsize=1)
def _load_customers() -> pd.DataFrame:
    """Load and cache the customers CSV. Called at most once per process."""
    try:
        df = pd.read_csv(_CUSTOMER_CSV)
        df.columns = df.columns.str.strip().str.lower()
        df["name_lower"] = df["name"].str.lower().str.strip()
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def look_up_customer(name: str) -> dict:
    """
    Look up a customer by name (case-insensitive, exact match first,
    then partial match).

    Returns a dict of signals or {} if not found.
    """
    df = _load_customers()
    if df.empty:
        return {}

    name_lower = name.strip().lower()

    # Exact match
    row = df[df["name_lower"] == name_lower]

    # Fallback: first name match
    if row.empty:
        first_name = name_lower.split()[0] if name_lower else ""
        row = df[df["name_lower"].str.startswith(first_name)] if first_name else pd.DataFrame()

    if row.empty:
        return {}

    r = row.iloc[0]
    return {
        "customer_id": str(r.get("customer_id", "")),
        "age": r.get("age", None),
        "gender": r.get("gender", ""),
        "city": r.get("city", ""),
        "income": r.get("income", None),
        "credit_score": r.get("credit_score", None),
        "loyalty_points": r.get("loyalty_points", 0),
        "avg_order_value": r.get("avg_order_value", None),
        "favorite_category": r.get("favorite_category", ""),
        "churn_risk": r.get("churn_risk", None),
        "subscription": r.get("subscription", False),
        "total_spent": r.get("total_spent", None),
        "device_type": r.get("device_type", ""),
    }


def build_enrichment_context(signals: dict) -> dict:
    """
    Convert raw customer signals into three usable artefacts:

    Returns
    -------
    {
        "score_bias": dict[str, float]   # passed to recommendation scorer
        "llm_snippet": str               # appended to LLM persona context
        "preferred_category": str | None # used as Pinecone filter hint
    }
    """
    if not signals:
        return {"score_bias": {}, "llm_snippet": "", "preferred_category": None}

    bias: dict[str, float] = {}
    snippets: list[str] = []

    # ── Income / credit → price sensitivity ──────────────────────────────
    income = signals.get("income")
    credit = signals.get("credit_score")
    avg_order = signals.get("avg_order_value")

    if income and income > 100_000:
        bias["premium"] = 0.10          # boost products with higher price points
        snippets.append("High-income customer — may prefer premium or branded products.")
    elif income and income < 40_000:
        bias["budget"] = 0.10           # subtle boost for lower-priced items
        snippets.append("Budget-conscious customer — highlight value-for-money options.")

    if credit and credit > 700:
        snippets.append("Excellent credit score — financially stable buyer.")

    # ── Loyalty ───────────────────────────────────────────────────────────
    loyalty = signals.get("loyalty_points", 0)
    if loyalty > 3000:
        snippets.append(f"High-loyalty customer with {int(loyalty)} points — likely a repeat buyer.")
        bias["loyalty"] = 0.05

    # ── Churn risk ────────────────────────────────────────────────────────
    churn = signals.get("churn_risk")
    if churn and churn > 0.6:
        snippets.append("High churn risk — recommend engaging, high-value products to retain them.")
        bias["engagement"] = 0.08

    # ── Favourite category ────────────────────────────────────────────────
    fav_cat = str(signals.get("favorite_category", "")).strip()
    preferred_category = fav_cat if fav_cat else None
    if fav_cat:
        snippets.append(f"Historically purchases most from: {fav_cat}.")

    # ── Subscription status ───────────────────────────────────────────────
    if signals.get("subscription"):
        snippets.append("Active subscriber — comfortable with recurring purchases.")

    llm_snippet = " ".join(snippets)
    return {
        "score_bias": bias,
        "llm_snippet": llm_snippet,
        "preferred_category": preferred_category,
    }
