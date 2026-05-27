"""
Sales Graph  (v2)
-----------------
LangGraph workflow with proper branching.

v1: persona_node → recommendation_node → END  (linear, no validation)

v2 graph:
    persona_node
         │
    validate_node ──(invalid)──→ error_node → END
         │ (valid)
    recommendation_node
         │
    diversity_check_node ──(too similar)──→ widen_query_node ──┐
         │ (diverse enough)                                      │
         └──────────────────────────────────────────────────────┘
         │
        END

New nodes:
- validate_node    : checks persona completeness, routes to error_node if bad.
- error_node       : populates a user-facing error message, terminates.
- diversity_check_node : checks if top results are all from the same category.
- widen_query_node : if diversity is poor, runs a secondary broad query and
                     merges results before final selection.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

from agents.persona_agent import build_persona_context, validate_persona
from agents.recommendation_agent import recommend_products


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class SalesState(TypedDict):
    persona: dict
    persona_context: str
    recommendations: list
    ai_explanation: str
    intents: list
    customer_signals: dict
    error: Optional[str]          # populated if validation fails
    validation_passed: bool


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def persona_node(state: SalesState) -> dict:
    """Build and store the persona context string."""
    context = build_persona_context(state["persona"])
    return {"persona_context": context, "error": None}


def validate_node(state: SalesState) -> dict:
    """Validate persona completeness. Sets validation_passed flag."""
    is_valid, error_msg = validate_persona(state["persona"])
    return {
        "validation_passed": is_valid,
        "error": error_msg if not is_valid else None,
    }


def error_node(state: SalesState) -> dict:
    """Terminal node for invalid personas — returns empty recommendations."""
    return {
        "recommendations": [],
        "ai_explanation": state.get("error", "Invalid persona — please fill in the required fields."),
        "intents": [],
        "customer_signals": {},
    }


def recommendation_node(state: SalesState) -> dict:
    """Core recommendation pipeline."""
    result = recommend_products(state["persona"])
    return result


def diversity_check_node(state: SalesState) -> dict:
    """
    Check whether results are diverse enough (≥2 distinct categories).
    If not, set a flag for the widen step — but don't change recommendations yet.
    """
    recs = state.get("recommendations", [])
    categories = {r.get("category", "") for r in recs}
    needs_widening = len(categories) < 2 and len(recs) > 0
    return {"_needs_widening": needs_widening}


def widen_query_node(state: SalesState) -> dict:
    """
    Secondary query with a broader persona (drop niche interests) to inject
    at least one product from a different category.
    Only runs when diversity_check flagged it.
    """
    if not state.get("_needs_widening", False):
        return {}

    # Build a generic persona to get cross-category suggestions
    broad_persona = {
        "name": state["persona"].get("name", ""),
        "age": state["persona"].get("age", ""),
        "income": state["persona"].get("income", ""),
        "interests": ["general", "popular", "bestseller"],
        "purchase_history": [],
    }
    broad_result = recommend_products(broad_persona)

    existing = state.get("recommendations", [])
    existing_names = {r["product"] for r in existing}

    # Add one product from a different category not already in list
    for product in broad_result.get("recommendations", []):
        if product["product"] not in existing_names:
            cat = product.get("category", "")
            existing_cats = {r.get("category", "") for r in existing}
            if cat not in existing_cats:
                existing = (existing + [product])[:5]
                break

    return {"recommendations": existing}


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_validation(state: SalesState) -> str:
    if state.get("validation_passed", False):
        return "recommendation_node"
    return "error_node"


def route_after_diversity(state: SalesState) -> str:
    if state.get("_needs_widening", False):
        return "widen_query_node"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

graph = StateGraph(SalesState)

graph.add_node("persona_node", persona_node)
graph.add_node("validate_node", validate_node)
graph.add_node("error_node", error_node)
graph.add_node("recommendation_node", recommendation_node)
graph.add_node("diversity_check_node", diversity_check_node)
graph.add_node("widen_query_node", widen_query_node)

graph.set_entry_point("persona_node")

graph.add_edge("persona_node", "validate_node")

graph.add_conditional_edges(
    "validate_node",
    route_after_validation,
    {
        "recommendation_node": "recommendation_node",
        "error_node": "error_node",
    },
)

graph.add_edge("error_node", END)
graph.add_edge("recommendation_node", "diversity_check_node")

graph.add_conditional_edges(
    "diversity_check_node",
    route_after_diversity,
    {
        "widen_query_node": "widen_query_node",
        END: END,
    },
)

graph.add_edge("widen_query_node", END)

app = graph.compile()
