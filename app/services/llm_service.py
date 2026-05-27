"""
LLM Service
-----------
Wraps the local Ollama / LLaMA3 instance.

Changes vs v1:
- Temperature lowered to 0.2 for more consistent, factual explanations.
- Added a system prompt that enforces structured output format so the
  recommendation_agent can rely on a predictable response shape.
- Exposed invoke_structured() for callers that need JSON-mode output.
"""

from langchain_community.llms import Ollama

# ---------------------------------------------------------------------------
# Primary LLM — used for recommendation explanations
# ---------------------------------------------------------------------------
llm = Ollama(
    model="llama3:latest",
    temperature=0.2,       # lower = more focused, less hallucination
    num_predict=512,       # cap tokens to keep explanations concise
)

# System prompt injected into every explanation request
EXPLANATION_SYSTEM_PROMPT = (
    "You are a helpful retail product recommendation assistant. "
    "You write concise, professional, and friendly product explanations. "
    "Never mention that you are an AI. "
    "Never refuse to discuss product categories — describe them in tasteful, "
    "professional retail language. "
    "Always respond in the exact structured format requested."
)


def build_explanation_prompt(persona_context: str, products_context: str, intent_hint: str = "") -> str:
    """
    Build the full prompt for the explanation step.

    Returns a string ready to pass to llm.invoke().
    """
    guidance = intent_hint or "Provide a balanced explanation covering relevance to the customer's interests and purchase history."

    return f"""{EXPLANATION_SYSTEM_PROMPT}

Customer Persona:
{persona_context}

Recommendation Guidance:
{guidance}

Recommended Products:
{products_context}

Instructions:
For each product, write exactly ONE sentence explaining why it suits this customer.
Format your response exactly like this (replace the example text):
1. [Product Name]: One sentence about fit.
2. [Product Name]: One sentence about fit.
...

End with a 1-sentence summary of the overall recommendation strategy for this customer.
Keep the total response under 200 words. Do not include any preamble or extra commentary."""
