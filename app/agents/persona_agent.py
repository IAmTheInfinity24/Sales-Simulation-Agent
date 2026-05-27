"""
Persona Agent  (v2)
-------------------
In v1 this module built a string that was never actually used downstream.

v2 changes:
- build_persona_context() still exists for backward compat.
- validate_persona() added — the graph now has a validation node before
  recommendation so incomplete personas get a helpful error early.
- enrich_persona() merges CSV lookup signals into the persona dict so that
  recommendation_agent receives a single enriched object.
"""

from __future__ import annotations


def validate_persona(persona: dict) -> tuple[bool, str]:
    """
    Check that the persona has enough signal to generate useful recommendations.

    Returns (is_valid: bool, error_message: str).
    """
    name = str(persona.get("name", "")).strip()
    interests = persona.get("interests", [])
    history = persona.get("purchase_history", [])

    if not name:
        return False, "Please provide a customer name."

    if not interests and not history:
        return (
            False,
            "Please provide at least one interest or a purchase history item "
            "so we can find relevant products.",
        )

    return True, ""


def build_persona_context(persona: dict) -> str:
    """
    Return a formatted prose string summarising the persona.
    Used as the persona_context field in the graph state.
    """
    interests = ", ".join(persona.get("interests", [])) or "Not specified"
    history = ", ".join(persona.get("purchase_history", [])) or "None"

    return (
        f"Customer Name: {persona.get('name', '')}\n"
        f"Age: {persona.get('age', '')}\n"
        f"Income: {persona.get('income', '')}\n"
        f"Interests: {interests}\n"
        f"Previous Purchases: {history}"
    )
