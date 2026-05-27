"""
Recommendation Agent  (v3)
--------------------------
Extends v2 with three additions required by the agentic graph:

1. override_embedding parameter
   recommend_products() now accepts an optional pre-built embedding vector
   (e.g. a preference-blended vector from router_agent.blend_embeddings).
   When provided it replaces the freshly generated query embedding so that
   preference-boosted retrieval works without changing any other logic.

2. _raw_scores in return value
   The function now returns a "_raw_scores" key containing the raw semantic
   similarity scores of candidates before re-ranking.  The confidence_check
   node uses these to decide whether a fallback is needed.

3. recommend_products_broad()
   A second entry point that runs the same pipeline but with a simplified
   persona (removes niche interests, raises top_k to 100).  Used by the
   fallback node when the primary retrieval confidence is too low.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from services.embedding_service import generate_embedding
from services.llm_service import llm, build_explanation_prompt
from services.pinecone_service import query_index
from utils.intent_classifier import (
    detect_intents,
    get_query_expansion,
    get_score_rules,
    get_intent_hint,
)
from utils.customer_enrichment import look_up_customer, build_enrichment_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K    = 60
BROAD_TOP_K        = 100
FINAL_TOP_N        = 5
MIN_CATEGORY_DIVERSITY = 2
MAX_SAME_BRAND     = 2
SEMANTIC_WEIGHT    = 0.60
KEYWORD_WEIGHT     = 0.40
MIN_RELEVANCE_SCORE = 0.25


# ---------------------------------------------------------------------------
# Internal helpers  (unchanged from v2)
# ---------------------------------------------------------------------------

def _normalise_terms(values) -> list[str]:
    terms: list[str] = []
    for value in values or []:
        chunk = str(value).lower()
        for token in re.split(r"[^a-z0-9]+", chunk):
            token = token.strip()
            if token:
                terms.append(token)
    return terms


def _normalise_product_name(name: str) -> str:
    text = re.sub(r"[^a-z0-9]", "", name.lower())
    for suffix in ("treat", "bar", "pack", "combo", "mini", "large", "small", "value"):
        text = text.replace(suffix, "")
    return text


def _names_are_similar(a: str, b: str, max_edits: int = 2) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_edits + 1:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    diffs = sum(1 for i, c in enumerate(shorter) if i < len(longer) and c != longer[i])
    diffs += len(longer) - len(shorter)
    return diffs <= max_edits


def _is_duplicate(product_name: str, brand: str, seen: list[tuple[str, str]]) -> bool:
    norm_new   = _normalise_product_name(product_name)
    brand_norm = re.sub(r"[^a-z0-9]", "", brand.lower())
    for seen_name, seen_brand in seen:
        if seen_brand == brand_norm and _names_are_similar(norm_new, seen_name):
            return True
    return False


def _build_query_text(persona: dict, intents: set[str]) -> str:
    age       = persona.get("age", "")
    interests = ", ".join(persona.get("interests", []))
    history   = ", ".join(persona.get("purchase_history", []))
    expansion = get_query_expansion(intents)

    parts = [f"A {age}-year-old customer" if age else "A customer"]
    if interests:
        parts.append(f"interested in {interests}")
    if history:
        parts.append(f"who previously purchased {history}")
    parts.append("looking for product recommendations.")
    if expansion:
        parts.append(expansion)

    return " ".join(parts)


def _sanitize_persona_for_llm(text: str) -> str:
    replacements = [
        (r"\bsex\b",      "intimate wellness"),
        (r"\bsexual\b",   "intimate wellness"),
        (r"\bcondoms?\b", "protective wellness products"),
    ]
    safe = str(text)
    for pattern, replacement in replacements:
        safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)
    return safe


def _get_text_blob(metadata: dict) -> str:
    return " ".join(
        str(metadata.get(field, "")).lower()
        for field in ("product", "category", "sub_category", "brand", "description", "type")
    )


def _compute_keyword_score(text_blob: str, persona_terms: list[str], score_rules: list) -> float:
    kw_score = 0.0
    matched  = sum(1 for term in persona_terms if term in text_blob)
    if matched:
        kw_score += min(matched / max(len(persona_terms), 1), 0.5)
    for terms, delta in score_rules:
        if any(term in text_blob for term in terms):
            kw_score += delta
    return max(0.0, min(1.0, kw_score))


def _compute_price_bias(product_data: dict, score_bias: dict) -> float:
    bias = 0.0
    try:
        price = float(str(product_data.get("sale_price", "0")).replace("₹", "").strip())
    except (ValueError, TypeError):
        return bias
    if "premium" in score_bias and price > 500:
        bias += score_bias["premium"]
    if "budget" in score_bias and price < 200:
        bias += score_bias["budget"]
    return bias


def _enforce_diversity(
    ranked:         list[dict],
    n:              int,
    min_categories: int,
    max_per_brand:  int,
) -> list[dict]:
    if not ranked:
        return []
    selected:        list[dict]       = []
    seen_categories: set[str]         = set()
    brand_counts:    dict[str, int]   = {}
    remaining = list(ranked)

    while len(selected) < n and remaining:
        placed = False
        for i, product in enumerate(remaining):
            cat   = str(product.get("category", "")).strip()
            brand = str(product.get("brand",    "")).strip().lower()

            needs_new_cat = len(seen_categories) < min_categories
            brand_ok      = brand_counts.get(brand, 0) < max_per_brand
            cat_ok        = (not needs_new_cat) or (cat not in seen_categories)

            if brand_ok and cat_ok:
                selected.append(product)
                seen_categories.add(cat)
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
                remaining.pop(i)
                placed = True
                break

        if not placed:
            selected.append(remaining.pop(0))

    return selected


def _fallback_explanation(recommendations: list[dict], intents: set[str]) -> str:
    intent_str   = ", ".join(sorted(intents)) if intents else "general interests"
    names        = [r["product"] for r in recommendations]
    product_list = "; ".join(names[:3])
    return (
        f"Based on your {intent_str}, we recommend: {product_list} and more. "
        "These products align with your purchase history and lifestyle preferences."
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    persona:           dict,
    top_k:             int             = RETRIEVAL_TOP_K,
    override_embedding: Optional[list[float]] = None,
) -> dict:
    """
    Shared pipeline used by both recommend_products() and recommend_products_broad().

    Parameters
    ----------
    persona            : customer persona dict
    top_k              : Pinecone candidates to fetch
    override_embedding : if provided, skip query embedding generation and use this
                         vector directly (e.g. a preference-blended embedding)
    """

    # ── 1. DETECT INTENTS ────────────────────────────────────────────────
    intents = detect_intents(persona)

    # ── 2. CUSTOMER ENRICHMENT ───────────────────────────────────────────
    customer_signals   = look_up_customer(persona.get("name", ""))
    enrichment         = build_enrichment_context(customer_signals)
    score_bias         = enrichment["score_bias"]
    llm_enrichment_snippet = enrichment["llm_snippet"]

    # ── 3. BUILD QUERY TEXT ───────────────────────────────────────────────
    query_text = _build_query_text(persona, intents)

    # ── 4. EMBEDDING  (use override if provided) ──────────────────────────
    if override_embedding is not None:
        embedding = override_embedding
        logger.debug("Using override (preference-blended) embedding")
    else:
        embedding = generate_embedding(query_text, is_query=True)

    # ── 5. RETRIEVE FROM PINECONE ─────────────────────────────────────────
    results = query_index(vector=embedding, top_k=top_k)

    # ── 6. DEDUPLICATE + BUILD CANDIDATE LIST ─────────────────────────────
    persona_terms = (
        _normalise_terms(persona.get("interests",       []))
        + _normalise_terms(persona.get("purchase_history", []))
    )
    score_rules = get_score_rules(intents)

    seen_products: list[tuple[str, str]] = []
    candidates:    list[dict]            = []
    raw_scores:    list[float]           = []

    for match in results.get("matches", []):
        metadata     = match.get("metadata", {})
        product_name = str(metadata.get("product", "Unknown Product")).strip()
        brand        = str(metadata.get("brand",   "")).strip()

        if _is_duplicate(product_name, brand, seen_products):
            continue
        seen_products.append((
            _normalise_product_name(product_name),
            re.sub(r"[^a-z0-9]", "", brand.lower()),
        ))

        text_blob      = _get_text_blob(metadata)
        semantic_score = float(match.get("score", 0.0))
        keyword_score  = _compute_keyword_score(text_blob, persona_terms, score_rules)
        price_bias     = _compute_price_bias(metadata, score_bias)

        final_score = (
            SEMANTIC_WEIGHT * semantic_score
            + KEYWORD_WEIGHT * keyword_score
            + price_bias
        )

        if final_score < MIN_RELEVANCE_SCORE:
            continue

        raw_scores.append(semantic_score)
        candidates.append({
            "_score"      : final_score,
            "product"     : product_name,
            "category"    : metadata.get("category",     "N/A"),
            "sub_category": metadata.get("sub_category", "N/A"),
            "brand"       : brand,
            "sale_price"  : metadata.get("sale_price",   "N/A"),
            "market_price": metadata.get("market_price", "N/A"),
            "rating"      : metadata.get("rating",       "N/A"),
            "description" : metadata.get("description",  ""),
        })

    # ── 7. RANK + DIVERSITY ───────────────────────────────────────────────
    candidates.sort(key=lambda x: x["_score"], reverse=True)
    recommendations = _enforce_diversity(
        candidates, FINAL_TOP_N, MIN_CATEGORY_DIVERSITY, MAX_SAME_BRAND
    )
    for rec in recommendations:
        rec.pop("_score", None)

    if not recommendations:
        return {
            "recommendations" : [],
            "ai_explanation"  : "No relevant products were found in the current catalog.",
            "intents"         : sorted(intents),
            "customer_signals": customer_signals,
            "_raw_scores"     : [],
        }

    # ── 8. BUILD LLM PROMPT ───────────────────────────────────────────────
    raw_persona_block = (
        f"Name: {persona.get('name', '')}\n"
        f"Age: {persona.get('age', '')}\n"
        f"Income: {persona.get('income', '')}\n"
        f"Interests: {', '.join(persona.get('interests', []))}\n"
        f"Purchase History: {', '.join(persona.get('purchase_history', []))}\n"
        f"{llm_enrichment_snippet}"
    )
    safe_persona_block = _sanitize_persona_for_llm(raw_persona_block)

    products_lines = []
    for i, p in enumerate(recommendations, 1):
        products_lines.append(
            f"{i}. {p['product']} by {p['brand']}\n"
            f"   Category: {p['category']} > {p['sub_category']}\n"
            f"   Price: ₹{p['sale_price']} | Rating: {p['rating']}\n"
            f"   {p['description'][:200]}"
        )
    products_block = "\n\n".join(products_lines)
    intent_hint    = get_intent_hint(intents)

    prompt = build_explanation_prompt(
        persona_context  = safe_persona_block,
        products_context = products_block,
        intent_hint      = intent_hint,
    )

    # ── 9. LLM CALL ───────────────────────────────────────────────────────
    try:
        llm_response     = llm.invoke(prompt)
        explanation_text = str(llm_response).strip()
        if not explanation_text:
            raise ValueError("Empty LLM response")
    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        explanation_text = _fallback_explanation(recommendations, intents)

    return {
        "recommendations" : recommendations,
        "ai_explanation"  : explanation_text,
        "intents"         : sorted(intents),
        "customer_signals": customer_signals,
        "_raw_scores"     : raw_scores,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def recommend_products(
    persona:            dict,
    override_embedding: Optional[list[float]] = None,
) -> dict:
    """
    Standard recommendation pipeline.

    Parameters
    ----------
    persona            : customer persona dict (name, age, income, interests, purchase_history)
    override_embedding : optional pre-built query vector (e.g. preference-blended)

    Returns
    -------
    {
        "recommendations" : list[dict],
        "ai_explanation"  : str,
        "intents"         : list[str],
        "customer_signals": dict,
        "_raw_scores"     : list[float],   # consumed by confidence_check_node
    }
    """
    return _run_pipeline(
        persona,
        top_k              = RETRIEVAL_TOP_K,
        override_embedding = override_embedding,
    )


def recommend_products_broad(persona: dict) -> dict:
    """
    Fallback pipeline with a relaxed persona and larger candidate pool.

    Strips niche interests down to generic terms and raises top_k to 100
    so that borderline-relevant products have a chance to surface.

    Used by fallback_node when primary retrieval confidence is too low.
    """
    broad_persona = {
        "name"            : persona.get("name", ""),
        "age"             : persona.get("age",  ""),
        "income"          : persona.get("income", ""),
        # Keep only the first interest (most general) + popular signal
        "interests"       : (persona.get("interests", [])[:1] or []) + ["popular", "bestseller"],
        "purchase_history": [],
    }
    return _run_pipeline(broad_persona, top_k=BROAD_TOP_K)
