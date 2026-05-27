"""
Router Agent
------------
Decides the retrieval strategy for each recommendation request.

This replaces the implicit "always run standard search" assumption in v2.
The router inspects the user's preference profile and the detected intents,
then picks the best strategy — making the graph dynamically adaptive rather
than following a fixed path.

Strategies
----------
"preference_boosted"
    The user has enough interaction history to have a meaningful preference
    embedding.  The query embedding is blended with the stored preference
    vector before Pinecone search so that semantically similar products to
    historically liked ones surface higher.

"standard"
    Normal query-embedding search.  Used for new users or those with
    insufficient feedback history.

Fallback strategies (chosen by confidence_check_node after retrieval)
---------------------------------------------------------------------
"none"            : results are good enough, no fallback needed
"broad"           : re-run with a generalised persona (fewer niche terms)
"category_popular": filter Pinecone by preferred categories + sort by rating
"""

from __future__ import annotations

import logging
from typing import Optional

from memory.user_store import load_profile

logger = logging.getLogger(__name__)

# Thresholds for promoting a user to preference_boosted strategy
_MIN_INTERACTIONS = 2
_MIN_LIKED        = 3

# Post-retrieval confidence thresholds
CONFIDENCE_OK      = 0.38   # above → results are fine
CONFIDENCE_BROAD   = 0.28   # between BROAD and OK → run broad fallback
# below CONFIDENCE_BROAD → run category_popular fallback


def decide_strategy(name: str, intents: set[str]) -> str:
    """
    Choose the initial retrieval strategy for this customer.

    Parameters
    ----------
    name    : customer name (used to load their profile)
    intents : set of active intents detected from the persona

    Returns
    -------
    "preference_boosted" | "standard"
    """
    profile = load_profile(name)
    if profile is None:
        return "standard"

    interactions  = profile.get("interaction_count", 0)
    liked_count   = len(profile.get("liked_products", []))
    has_embedding = profile.get("preference_embedding") is not None

    if interactions >= _MIN_INTERACTIONS and liked_count >= _MIN_LIKED and has_embedding:
        logger.debug(
            "Router → preference_boosted for '%s' (interactions=%d, liked=%d)",
            name, interactions, liked_count,
        )
        return "preference_boosted"

    logger.debug("Router → standard for '%s'", name)
    return "standard"


def decide_fallback_strategy(confidence: float, result_count: int) -> str:
    """
    Decide whether and how to fall back after a low-confidence retrieval.

    Parameters
    ----------
    confidence   : average semantic score of top results (0–1)
    result_count : number of results returned

    Returns
    -------
    "none" | "broad" | "category_popular"
    """
    # Fewer than 3 results is always treated as low confidence
    if result_count < 3:
        confidence = min(confidence, 0.20)

    if confidence >= CONFIDENCE_OK:
        return "none"
    if confidence >= CONFIDENCE_BROAD:
        return "broad"
    return "category_popular"


def blend_embeddings(
    query_embedding: list[float],
    preference_embedding: list[float],
    preference_weight: float = 0.25,
) -> list[float]:
    """
    Linearly interpolate between the current query embedding and the
    user's stored preference embedding, then re-normalise to unit length.

    preference_weight=0.25 → 75% current query, 25% historical taste.
    Keeping the weight low ensures the query remains dominant so that a
    query for 'helmets' doesn't get dragged toward 'snacks' just because
    the user liked snacks in the past.
    """
    if len(query_embedding) != len(preference_embedding):
        logger.warning("Embedding dimension mismatch — skipping blend")
        return query_embedding

    blended = [
        (1.0 - preference_weight) * q + preference_weight * p
        for q, p in zip(query_embedding, preference_embedding)
    ]
    norm = (sum(x * x for x in blended) ** 0.5) or 1.0
    return [x / norm for x in blended]
