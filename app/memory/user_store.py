"""
User Profile Store
------------------
Persists user preference profiles to disk (JSON files under data/user_profiles/).
In production, swap the load/save functions for a PostgreSQL or DynamoDB client
without touching any other code.

Profile schema
--------------
{
    "user_id"            : str,           # MD5 hash of lowercased name
    "name"               : str,
    "interaction_count"  : int,
    "liked_categories"   : {str: int},    # category → cumulative score
    "liked_products"     : [str],         # product names user approved
    "disliked_products"  : [str],
    "preference_embedding": [float] | None,  # EMA-blended embedding
    "last_updated"       : str,           # ISO timestamp
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "user_profiles")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    os.makedirs(_PROFILES_DIR, exist_ok=True)


def _user_id(name: str) -> str:
    return hashlib.md5(name.strip().lower().encode()).hexdigest()[:12]


def _profile_path(name: str) -> str:
    _ensure_dir()
    return os.path.join(_PROFILES_DIR, f"{_user_id(name)}.json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_profile(name: str) -> Optional[dict]:
    """Load a user profile from disk. Returns None if not found."""
    path = _profile_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load profile for '%s': %s", name, exc)
        return None


def save_profile(profile: dict) -> bool:
    """Persist a user profile. Returns True on success."""
    name = profile.get("name", "")
    if not name:
        return False
    path = _profile_path(name)
    try:
        profile["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        return True
    except Exception as exc:
        logger.warning("Failed to save profile for '%s': %s", name, exc)
        return False


def create_empty_profile(name: str) -> dict:
    """Return a blank profile for a first-time user."""
    return {
        "user_id": _user_id(name),
        "name": name.strip().lower(),
        "interaction_count": 0,
        "liked_categories": {},
        "liked_products": [],
        "disliked_products": [],
        "preference_embedding": None,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def update_profile_with_feedback(
    name: str,
    liked: list[dict],
    disliked: list[dict],
) -> dict:
    """
    Merge a round of thumbs-up / thumbs-down feedback into the profile.

    Parameters
    ----------
    name     : customer name
    liked    : list of product dicts the user approved
    disliked : list of product dicts the user rejected

    Returns the saved profile dict.
    """
    profile = load_profile(name) or create_empty_profile(name)

    liked_set    = set(profile.get("liked_products",    []))
    disliked_set = set(profile.get("disliked_products", []))
    cat_scores   = dict(profile.get("liked_categories", {}))

    for p in liked:
        pname = p.get("product", "")
        if pname:
            liked_set.add(pname)
            disliked_set.discard(pname)       # un-dislike if previously disliked
        cat = p.get("category", "")
        if cat:
            cat_scores[cat] = cat_scores.get(cat, 0) + 1

    for p in disliked:
        pname = p.get("product", "")
        if pname:
            disliked_set.add(pname)
            liked_set.discard(pname)
        cat = p.get("category", "")
        if cat:
            cat_scores[cat] = max(0, cat_scores.get(cat, 0) - 1)

    profile["liked_products"]    = list(liked_set)[:200]   # cap unbounded growth
    profile["disliked_products"] = list(disliked_set)[:200]
    profile["liked_categories"]  = cat_scores
    profile["interaction_count"] = profile.get("interaction_count", 0) + 1

    save_profile(profile)
    return profile


def update_preference_embedding(
    name: str,
    new_embedding: list[float],
    weight: float = 0.30,
) -> dict:
    """
    Blend a new signal embedding into the stored preference embedding
    using an Exponential Moving Average.

    weight=0.30 → new signal contributes 30%, prior history 70%.

    Returns the updated profile.
    """
    profile = load_profile(name) or create_empty_profile(name)
    existing = profile.get("preference_embedding")

    if existing is None or len(existing) != len(new_embedding):
        profile["preference_embedding"] = new_embedding
    else:
        blended = [
            weight * n + (1.0 - weight) * e
            for n, e in zip(new_embedding, existing)
        ]
        # Re-normalise to unit sphere so cosine sim == dot product
        norm = (sum(x * x for x in blended) ** 0.5) or 1.0
        profile["preference_embedding"] = [x / norm for x in blended]

    save_profile(profile)
    return profile


def get_preferred_categories(name: str, top_n: int = 3) -> list[str]:
    """Return the user's top N categories ranked by cumulative like score."""
    profile = load_profile(name)
    if not profile:
        return []
    cats = profile.get("liked_categories", {})
    return sorted((c for c, s in cats.items() if s > 0), key=cats.get, reverse=True)[:top_n]


def profile_summary(name: str) -> Optional[dict]:
    """
    Return a lightweight summary dict for the UI.
    Returns None if no profile exists.
    """
    p = load_profile(name)
    if not p:
        return None
    return {
        "interaction_count" : p.get("interaction_count", 0),
        "liked_count"       : len(p.get("liked_products", [])),
        "top_categories"    : get_preferred_categories(name, top_n=3),
        "has_embedding"     : p.get("preference_embedding") is not None,
        "last_updated"      : p.get("last_updated", ""),
    }
