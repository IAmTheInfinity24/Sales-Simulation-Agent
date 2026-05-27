"""
Agentic Graph  (v3)
-------------------
Replaces the fixed pipeline in sales_graph.py with a memory-aware,
dynamically-routed workflow.  sales_graph.py is left untouched for
backward compatibility.

New capabilities vs v2
----------------------
1. memory_retrieve_node   — loads the user's preference profile before retrieval
2. dynamic_router_node    — chooses "preference_boosted" vs "standard" strategy
                            and pre-blends the query embedding when appropriate
3. confidence_check_node  — scores result quality; routes to fallback if low
4. fallback_node          — tiered: broad search → category_popular → graceful empty
5. Feedback is processed externally (by the UI) and stored via feedback_agent;
   the graph reads the resulting profile on the NEXT invocation

Graph topology
--------------
    persona_node
        → validate_node
            → (invalid)  error_node → END
            → (valid)    memory_retrieve_node
                → dynamic_router_node
                    → recommendation_node
                        → confidence_check_node
                            → (low)  fallback_node → diversity_check_node
                            → (ok)   diversity_check_node
                                → (needs widening)  widen_query_node → END
                                → (diverse enough)  END
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.persona_agent import build_persona_context, validate_persona
from agents.recommendation_agent import recommend_products, recommend_products_broad
from agents.router_agent import (
    blend_embeddings,
    decide_fallback_strategy,
    decide_strategy,
)
from memory.user_store import get_preferred_categories, load_profile
from services.embedding_service import generate_embedding
from services.pinecone_service import query_index
from utils.intent_classifier import detect_intents

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgenticState(TypedDict):
    # Input
    persona: dict

    # Persona processing
    persona_context:   str
    validation_passed: bool
    error:             Optional[str]

    # Memory
    user_profile:        Optional[dict]
    retrieval_strategy:  str                   # "standard" | "preference_boosted"
    preference_embedding: Optional[list[float]]

    # Retrieval results
    intents:          list
    recommendations:  list
    ai_explanation:   str
    customer_signals: dict

    # Confidence + fallback control
    confidence_score:  float
    fallback_strategy: str        # "none" | "broad" | "category_popular"
    fallback_applied:  bool

    # Diversity control
    _needs_widening: bool


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def persona_node(state: AgenticState) -> dict:
    context = build_persona_context(state["persona"])
    return {"persona_context": context, "error": None}


def validate_node(state: AgenticState) -> dict:
    is_valid, error_msg = validate_persona(state["persona"])
    return {
        "validation_passed": is_valid,
        "error": error_msg if not is_valid else None,
    }


def error_node(state: AgenticState) -> dict:
    return {
        "recommendations" : [],
        "ai_explanation"  : state.get("error", "Invalid persona — please fill in the required fields."),
        "intents"         : [],
        "customer_signals": {},
        "confidence_score": 0.0,
        "fallback_applied": False,
        "retrieval_strategy": "none",
    }


def memory_retrieve_node(state: AgenticState) -> dict:
    """Load the user's preference profile from the local store."""
    name    = state["persona"].get("name", "")
    profile = load_profile(name)
    pref_emb = profile.get("preference_embedding") if profile else None
    logger.debug(
        "Memory retrieve for '%s': profile_found=%s, has_embedding=%s",
        name, profile is not None, pref_emb is not None,
    )
    return {
        "user_profile"        : profile,
        "preference_embedding": pref_emb,
    }


def dynamic_router_node(state: AgenticState) -> dict:
    """
    Decide retrieval strategy.  If preference_boosted is chosen, pre-blend
    the fresh query embedding with the stored preference vector and store
    the result so recommendation_node can use it directly.
    """
    name    = state["persona"].get("name", "")
    intents = set(detect_intents(state["persona"]))
    strategy = decide_strategy(name, intents)

    blended_embedding: Optional[list[float]] = None

    if strategy == "preference_boosted":
        pref_emb = state.get("preference_embedding")
        if pref_emb:
            from agents.recommendation_agent import _build_query_text
            query_text = _build_query_text(state["persona"], intents)
            query_emb  = generate_embedding(query_text, is_query=True)
            blended_embedding = blend_embeddings(query_emb, pref_emb, preference_weight=0.25)
            logger.debug("preference_boosted embedding blended for '%s'", name)
        else:
            strategy = "standard"   # no stored embedding yet — fall back

    return {
        "retrieval_strategy" : strategy,
        "preference_embedding": blended_embedding,  # None for standard
    }


def recommendation_node(state: AgenticState) -> dict:
    """Core recommendation pipeline — passes blended embedding when available."""
    result = recommend_products(
        state["persona"],
        override_embedding=state.get("preference_embedding"),
    )

    # Compute confidence from raw scores
    raw_scores      = result.pop("_raw_scores", [])
    top5_scores     = raw_scores[:5]
    confidence      = sum(top5_scores) / len(top5_scores) if top5_scores else 0.0

    result["confidence_score"] = confidence
    result["fallback_applied"] = False
    return result


def confidence_check_node(state: AgenticState) -> dict:
    """
    Evaluate result quality.  A low result count is treated as low confidence
    regardless of the score.
    """
    confidence   = state.get("confidence_score", 0.0)
    result_count = len(state.get("recommendations", []))
    fallback     = decide_fallback_strategy(confidence, result_count)
    logger.debug(
        "Confidence check: score=%.3f, results=%d → fallback=%s",
        confidence, result_count, fallback,
    )
    return {"fallback_strategy": fallback}


def fallback_node(state: AgenticState) -> dict:
    """
    Tiered fallback:
      broad          → re-run with simplified persona + larger top_k
      category_popular → filter Pinecone by user's preferred categories
    """
    strategy = state.get("fallback_strategy", "none")
    persona  = state["persona"]

    if strategy == "broad":
        broad_result = recommend_products_broad(persona)
        recs = broad_result.get("recommendations", [])
        if recs:
            logger.debug("Broad fallback produced %d results", len(recs))
            return {
                "recommendations" : recs,
                "ai_explanation"  : broad_result.get("ai_explanation", ""),
                "fallback_applied": True,
            }
        # If broad also came up short, escalate to category_popular
        strategy = "category_popular"

    if strategy == "category_popular":
        preferred = get_preferred_categories(persona.get("name", ""), top_n=2)
        cat_recs  = _category_popular_fallback(persona, preferred)
        if cat_recs:
            logger.debug("Category-popular fallback produced %d results", len(cat_recs))
            return {
                "recommendations" : cat_recs,
                "ai_explanation"  : (
                    "These are top-rated products from your favourite categories, "
                    "selected because your primary search returned limited results."
                ),
                "fallback_applied": True,
            }

    # All fallbacks exhausted — return original (possibly empty) results
    return {"fallback_applied": False}


def diversity_check_node(state: AgenticState) -> dict:
    recs       = state.get("recommendations", [])
    categories = {r.get("category", "") for r in recs}
    needs_widening = len(categories) < 2 and len(recs) > 0
    return {"_needs_widening": needs_widening}


def widen_query_node(state: AgenticState) -> dict:
    """Inject one product from a new category when results are too homogeneous."""
    if not state.get("_needs_widening", False):
        return {}

    broad_persona = {
        "name"            : state["persona"].get("name", ""),
        "age"             : state["persona"].get("age",  ""),
        "income"          : state["persona"].get("income", ""),
        "interests"       : ["general", "popular", "bestseller"],
        "purchase_history": [],
    }
    broad_result   = recommend_products(broad_persona)
    existing       = state.get("recommendations", [])
    existing_names = {r["product"] for r in existing}

    for product in broad_result.get("recommendations", []):
        if product["product"] not in existing_names:
            cat            = product.get("category", "")
            existing_cats  = {r.get("category", "") for r in existing}
            if cat not in existing_cats:
                existing = (existing + [product])[:5]
                break

    return {"recommendations": existing}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_validation(state: AgenticState) -> str:
    return "memory_retrieve_node" if state.get("validation_passed") else "error_node"


def route_after_confidence(state: AgenticState) -> str:
    return (
        "fallback_node"
        if state.get("fallback_strategy", "none") != "none"
        else "diversity_check_node"
    )


def route_after_diversity(state: AgenticState) -> str:
    return "widen_query_node" if state.get("_needs_widening", False) else END


# ---------------------------------------------------------------------------
# Category-popular fallback helper
# ---------------------------------------------------------------------------

def _category_popular_fallback(persona: dict, preferred_categories: list[str]) -> list[dict]:
    """
    Run a Pinecone query filtered to preferred categories.
    Falls back to an unfiltered popularity query if no preferred categories exist.
    """
    from agents.recommendation_agent import _build_query_text

    intents    = set(detect_intents(persona))
    query_text = _build_query_text(persona, intents)
    embedding  = generate_embedding(query_text, is_query=True)

    cats_to_try = preferred_categories if preferred_categories else [None]  # None = no filter

    for cat in cats_to_try:
        try:
            filter_dict = {"category": {"$eq": cat}} if cat else None
            results     = query_index(vector=embedding, top_k=20, filter_dict=filter_dict)
            matches     = results.get("matches", [])
            if matches:
                products = []
                for m in matches[:5]:
                    md = m.get("metadata", {})
                    products.append({
                        "product"     : md.get("product",     ""),
                        "brand"       : md.get("brand",       ""),
                        "category"    : md.get("category",    ""),
                        "sub_category": md.get("sub_category",""),
                        "sale_price"  : md.get("sale_price",  ""),
                        "market_price": md.get("market_price",""),
                        "rating"      : md.get("rating",      ""),
                        "description" : md.get("description", ""),
                    })
                if products:
                    return products
        except Exception as exc:
            logger.warning("Category fallback failed for '%s': %s", cat, exc)

    return []


# ---------------------------------------------------------------------------
# Build the compiled graph
# ---------------------------------------------------------------------------

graph = StateGraph(AgenticState)

graph.add_node("persona_node",          persona_node)
graph.add_node("validate_node",         validate_node)
graph.add_node("error_node",            error_node)
graph.add_node("memory_retrieve_node",  memory_retrieve_node)
graph.add_node("dynamic_router_node",   dynamic_router_node)
graph.add_node("recommendation_node",   recommendation_node)
graph.add_node("confidence_check_node", confidence_check_node)
graph.add_node("fallback_node",         fallback_node)
graph.add_node("diversity_check_node",  diversity_check_node)
graph.add_node("widen_query_node",      widen_query_node)

graph.set_entry_point("persona_node")

graph.add_edge("persona_node", "validate_node")

graph.add_conditional_edges(
    "validate_node",
    route_after_validation,
    {
        "memory_retrieve_node": "memory_retrieve_node",
        "error_node":           "error_node",
    },
)

graph.add_edge("error_node",           END)
graph.add_edge("memory_retrieve_node", "dynamic_router_node")
graph.add_edge("dynamic_router_node",  "recommendation_node")
graph.add_edge("recommendation_node",  "confidence_check_node")

graph.add_conditional_edges(
    "confidence_check_node",
    route_after_confidence,
    {
        "fallback_node":        "fallback_node",
        "diversity_check_node": "diversity_check_node",
    },
)

graph.add_edge("fallback_node", "diversity_check_node")

graph.add_conditional_edges(
    "diversity_check_node",
    route_after_diversity,
    {
        "widen_query_node": "widen_query_node",
        END:                END,
    },
)

graph.add_edge("widen_query_node", END)

app = graph.compile()
