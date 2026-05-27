"""
Intent Classifier
-----------------
Replaces the brittle hardcoded keyword lists in v1 with a lightweight,
extensible intent detection system.

Design:
- INTENT_TAXONOMY defines intents as (intent_name → keyword_set) mappings.
- detect_intents() returns a set of active intents for a persona.
- get_query_expansion() returns domain-specific terms to append to the
  embedding query for each active intent (boosts recall on niche categories).
- get_score_rules() returns scoring rules (field→terms→boost) per intent,
  used by the weighted scoring function in recommendation_agent.py.
- get_intent_hint() returns a prose hint for the LLM prompt per intent set.

To add a new intent, add one entry to INTENT_TAXONOMY and optionally to
QUERY_EXPANSIONS, SCORE_RULES, and INTENT_HINTS.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Taxonomy — maps intent name → trigger keywords
# ---------------------------------------------------------------------------
INTENT_TAXONOMY: dict[str, set[str]] = {
    "fitness": {
        "fitness", "workout", "gym", "protein", "health", "healthy",
        "oats", "nutrition", "muscle", "exercise", "sports", "running",
        "cycling", "yoga", "crossfit", "whey", "creatine",
    },
    "travel": {
        "travel", "travelling", "traveling", "trip", "tour", "touring",
        "backpacking", "trekking", "hiking", "adventure", "vacation",
        "journey", "road trip",
    },
    "riding": {
        "ride", "riding", "bike", "motorcycle", "biker", "motorbike",
        "two-wheeler", "scooter", "cycling",
    },
    "sexual_wellness": {
        "condom", "condoms", "sexual", "sex", "intimacy", "lube",
        "lubricant", "intimate", "contraceptive", "protection",
    },
    "beauty": {
        "beauty", "skincare", "makeup", "cosmetic", "hair care",
        "grooming", "moisturizer", "serum", "sunscreen",
    },
    "baby": {
        "baby", "infant", "toddler", "newborn", "diaper", "nappy",
        "stroller", "feeding", "formula",
    },
    "home": {
        "home", "kitchen", "cooking", "cleaning", "household",
        "furniture", "decor", "storage", "organise", "organize",
    },
    "electronics": {
        "phone", "mobile", "laptop", "computer", "gadget", "electronics",
        "headphone", "earphone", "speaker", "camera", "charger",
    },
    "books": {
        "book", "books", "reading", "novel", "study", "learning",
        "education", "academic",
    },
    "pet": {
        "pet", "dog", "cat", "bird", "fish", "aquarium",
        "pet food", "pet care",
    },
    # ── NEW: snacks & grocery ─────────────────────────────────────
    "snacks": {
        "snack", "snacks", "namkeen", "biscuit", "biscuits", "chips",
        "crackers", "cookies", "wafers", "popcorn", "mixture", "bhujia",
        "mathri", "sev", "murukku", "chiwda", "fryums", "kurkure",
        "lays", "haldiram", "evening snack", "munchies", "crisps",
        "dry snack", "fried snack", "roasted snack", "puffed snack",
    },
    "beverages": {
        "chai", "tea", "coffee", "drink", "drinks", "juice", "beverage",
        "beverages", "cold drink", "soda", "water", "squash", "shake",
        "smoothie", "lemonade", "lassi", "buttermilk", "coconut water",
        "energy drink", "green tea", "black tea", "milk", "horlicks",
    },
    "grocery": {
        "grocery", "groceries", "dal", "rice", "flour", "atta", "maida",
        "spice", "spices", "masala", "oil", "ghee", "sugar", "salt",
        "pulses", "lentil", "cereal", "bread", "jam", "honey",
        "sauce", "ketchup", "pickle", "papad", "chutney", "instant",
        "ready to eat", "packaged food", "staple",
    },
}

# ---------------------------------------------------------------------------
# Query expansions — extra terms appended to embedding query per intent
# Boosts recall for niche categories that may use different vocabulary
# ---------------------------------------------------------------------------
QUERY_EXPANSIONS: dict[str, str] = {
    "fitness": "fitness protein bars healthy snacks sports nutrition energy supplements",
    "travel": "travel kit bottle flask sanitizer backpack on-the-go accessories travel essentials",
    "riding": "helmet motorcycle accessories riding gear bike safety visor",
    "sexual_wellness": "sexual wellness intimate care protection lubricants condoms",
    "beauty": "skincare hair care grooming beauty cosmetics moisturiser",
    "baby": "baby care infant products feeding accessories diaper",
    "home": "kitchen household cleaning home decor storage organiser",
    "electronics": "electronics gadgets mobile accessories charger earphones",
    "books": "books stationery learning educational",
    "pet": "pet food pet care accessories grooming treats",
    # ── NEW ──────────────────────────────────────────────────────
    "snacks": (
        "namkeen biscuits chips cookies crackers wafers sev bhujia mixture "
        "haldiram balaji kurkure mathri chiwda fryums munchies dry snacks "
        "fried snacks roasted peanuts popcorn evening snacks crunchy snacks"
    ),
    "beverages": (
        "chai tea coffee green tea black tea herbal tea instant tea "
        "cold drink juice lemonade lassi buttermilk coconut water "
        "energy drink milk shake health drink horlicks bournvita"
    ),
    "grocery": (
        "grocery dal rice atta flour spices masala oil ghee sugar salt "
        "pulses lentil cereal bread jam honey sauce ketchup pickle papad "
        "ready to eat packaged food instant noodles staples"
    ),
}

# ---------------------------------------------------------------------------
# Score rules — applied in the weighted scorer
# Format: {intent: [(field_terms_to_match, boost_delta), ...]}
# ---------------------------------------------------------------------------
SCORE_RULES: dict[str, list[tuple[list[str], float]]] = {
    "fitness": [
        (["fitness", "protein", "oats", "healthy", "nutrition", "sports", "energy"], 0.20),
    ],
    "travel": [
        (["travel", "bottle", "flask", "sanitizer", "backpack", "on-the-go", "travel kit"], 0.25),
        (["ayurveda", "ayurvedic", "kathi", "remedy", "kerala"], -0.15),
    ],
    "riding": [
        (["helmet", "motorcycle", "riding gear", "visor", "bike"], 0.30),
        (["travel", "bottle", "kit", "backpack"], 0.10),
    ],
    "sexual_wellness": [
        (["condom", "condoms", "sexual wellness", "intimate", "lube", "lubricant"], 0.30),
    ],
    "beauty": [
        (["skincare", "hair care", "grooming", "beauty", "cosmetic", "serum"], 0.20),
    ],
    "baby": [
        (["baby", "infant", "toddler", "diaper", "feeding"], 0.25),
    ],
    "home": [
        (["kitchen", "household", "cleaning", "storage", "organiser"], 0.15),
    ],
    "electronics": [
        (["electronics", "gadget", "mobile", "charger", "earphone", "speaker"], 0.20),
    ],
    # ── NEW ──────────────────────────────────────────────────────
    "snacks": [
        (["namkeen", "bhujia", "sev", "mixture", "chiwda", "mathri", "murukku"], 0.35),
        (["biscuit", "cookie", "cracker", "wafer", "chips", "crisps"], 0.30),
        (["popcorn", "fryums", "kurkure", "munchies", "roasted", "puffed"], 0.25),
        (["haldiram", "balaji", "bikaji", "parle", "britannia", "mcvitie"], 0.20),
        # demote completely unrelated categories
        (["shampoo", "helmet", "mobile", "laptop", "diaper", "medicine"], -0.30),
    ],
    "beverages": [
        (["chai", "tea", "coffee", "green tea", "black tea", "herbal tea"], 0.35),
        (["juice", "drink", "beverage", "lemonade", "lassi", "buttermilk"], 0.25),
        (["horlicks", "bournvita", "milo", "complan", "ovaltine"], 0.20),
        # demote unrelated
        (["helmet", "shampoo", "diaper", "laptop", "charger"], -0.30),
    ],
    "grocery": [
        (["dal", "rice", "atta", "flour", "spice", "masala", "oil", "ghee"], 0.25),
        (["pickle", "papad", "chutney", "sauce", "ketchup", "jam", "honey"], 0.20),
        (["instant", "ready to eat", "packaged", "cereal", "bread"], 0.15),
    ],
}

# ---------------------------------------------------------------------------
# LLM intent hints — one per intent, injected into the explanation prompt
# ---------------------------------------------------------------------------
INTENT_HINTS: dict[str, str] = {
    "fitness": (
        "Focus on how each product supports the customer's active lifestyle, "
        "training goals, nutritional needs, or performance improvement."
    ),
    "travel": (
        "Emphasise portability, convenience, and utility on-the-go. "
        "Highlight how each product makes travel easier or more comfortable."
    ),
    "riding": (
        "Emphasise safety, durability, and riding convenience. "
        "Prioritise helmets and protective gear; frame others as ride-companion items."
    ),
    "sexual_wellness": (
        "Use professional, tasteful retail language. "
        "Frame products around health, comfort, and personal wellness."
    ),
    "beauty": (
        "Focus on how each product enhances the customer's grooming or skincare routine."
    ),
    "baby": (
        "Focus on safety, comfort, and developmental appropriateness for infants/toddlers."
    ),
    "home": (
        "Focus on how each product improves the customer's home organisation, "
        "cooking experience, or household efficiency."
    ),
    "electronics": (
        "Highlight connectivity, performance, and compatibility with the customer's "
        "existing devices or tech lifestyle."
    ),
    # ── NEW ──────────────────────────────────────────────────────
    "snacks": (
        "Focus on taste, crunch, flavour variety, and snacking occasion (evening snacks, "
        "tea-time, on-the-go). Connect each product to the customer's preference for "
        "namkeen, biscuits, or similar savoury/sweet snack categories."
    ),
    "beverages": (
        "Highlight flavour, warmth, refreshment, and occasion (morning chai, afternoon "
        "tea, post-meal drink). Connect each beverage to the customer's chai or tea habit."
    ),
    "grocery": (
        "Focus on quality, brand trust, everyday utility, and value for money. "
        "Frame as reliable pantry staples the customer would repurchase regularly."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, return set of tokens."""
    import re
    tokens = set()
    for token in re.split(r"[^a-z0-9]+", text.lower()):
        token = token.strip()
        if token:
            tokens.add(token)
    return tokens


def detect_intents(persona: dict) -> set[str]:
    """
    Return the set of active intents for a persona.

    Checks interests, purchase_history, and name fields.
    """
    raw_parts = (
        list(persona.get("interests", []))
        + list(persona.get("purchase_history", []))
        + [str(persona.get("name", ""))]
    )
    tokens = _tokenise(" ".join(str(p) for p in raw_parts))

    # Also check multi-word phrases (e.g. "road trip")
    full_text = " ".join(str(p) for p in raw_parts).lower()

    active: set[str] = set()
    for intent, keywords in INTENT_TAXONOMY.items():
        for kw in keywords:
            if " " in kw:
                if kw in full_text:
                    active.add(intent)
                    break
            elif kw in tokens:
                active.add(intent)
                break

    return active


def get_query_expansion(intents: set[str]) -> str:
    """Return concatenated expansion text for all active intents."""
    parts = [QUERY_EXPANSIONS[i] for i in intents if i in QUERY_EXPANSIONS]
    return " ".join(parts)


def get_score_rules(intents: set[str]) -> list[tuple[list[str], float]]:
    """Return flat list of (terms, boost) rules for all active intents."""
    rules: list[tuple[list[str], float]] = []
    for intent in intents:
        rules.extend(SCORE_RULES.get(intent, []))
    return rules


def get_intent_hint(intents: set[str]) -> str:
    """Return a combined LLM prompt hint for the active intents."""
    hints = [INTENT_HINTS[i] for i in intents if i in INTENT_HINTS]
    return " ".join(hints)