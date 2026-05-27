"""
Feedback Agent
--------------
Processes user thumbs-up / thumbs-down signals collected by the UI and
persists them in the user preference profile.

Two effects per feedback round
-------------------------------
1. Category preference weights  — liked categories gain score, disliked lose.
   Used by the router to decide strategy on the next session.
2. Preference embedding update  — the embeddings of liked product descriptions
   are averaged and blended (EMA, weight=0.30) into the stored embedding.
   Used by blend_embeddings() for preference-boosted retrieval.
"""

from __future__ import annotations

import logging

from memory.user_store import update_profile_with_feedback, update_preference_embedding
from services.embedding_service import generate_embedding

logger = logging.getLogger(__name__)


def process_feedback(
    name: str,
    all_products: list[dict],
    feedback: dict[str, str],    # {product_name: "like" | "dislike"}
) -> dict:
    """
    Apply a round of user feedback to the stored profile.

    Parameters
    ----------
    name         : customer name
    all_products : full list of recommended products shown to the user
    feedback     : mapping from product name to "like" or "dislike"

    Returns
    -------
    {
        "updated"  : bool,
        "liked"    : list[str],   # product names that were liked
        "disliked" : list[str],
    }
    """
    if not feedback:
        return {"updated": False, "liked": [], "disliked": []}

    liked:    list[dict] = []
    disliked: list[dict] = []

    for product in all_products:
        pname  = product.get("product", "")
        signal = feedback.get(pname)
        if signal == "like":
            liked.append(product)
        elif signal == "dislike":
            disliked.append(product)

    if not liked and not disliked:
        return {"updated": False, "liked": [], "disliked": []}

    # Persist category preferences + product lists
    update_profile_with_feedback(name, liked, disliked)

    # Update preference embedding from liked products
    if liked:
        _update_embedding_from_liked(name, liked)

    summary = {
        "updated"  : True,
        "liked"    : [p["product"] for p in liked],
        "disliked" : [p["product"] for p in disliked],
    }
    logger.debug("Feedback processed for '%s': %s", name, summary)
    return summary


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _update_embedding_from_liked(name: str, liked_products: list[dict]) -> None:
    """
    Derive a signal embedding from liked product text, average across all
    liked products, then EMA-blend into the stored preference embedding.
    """
    texts: list[str] = []
    for p in liked_products:
        parts = [
            p.get("product",     ""),
            p.get("category",    ""),
            p.get("sub_category",""),
            p.get("description", "")[:300],
        ]
        text = " ".join(x for x in parts if x)
        if text.strip():
            texts.append(text)

    if not texts:
        return

    # Embed all liked-product texts (is_query=False → no BGE prefix)
    embeddings = [generate_embedding(t, is_query=False) for t in texts]

    # Average across all liked products
    n   = len(embeddings)
    avg = [sum(col) / n for col in zip(*embeddings)]

    # Normalise
    norm = (sum(x * x for x in avg) ** 0.5) or 1.0
    avg  = [x / norm for x in avg]

    update_preference_embedding(name, avg, weight=0.30)
    logger.debug(
        "Preference embedding updated for '%s' from %d liked products", name, n
    )
