"""
CLI test harness for the agentic recommendation graph.

Run:
    python app/main.py
    python app/main.py --name "Rahul" --age 29 --income "12 LPA" --interests "fitness,healthy food" --history "protein bars,almond milk,oats"
"""

import argparse
import json

from graph.agentic_graph import app


def run(persona: dict) -> None:
    print("\n== Persona ==")
    print(json.dumps(persona, indent=2))
    print("\n== Running agentic recommendation pipeline ==")

    response = app.invoke({"persona": persona})

    print(f"\n== Detected Intents: {response.get('intents', [])} ==")
    print(f"Retrieval strategy : {response.get('retrieval_strategy', 'N/A')}")
    print(f"Confidence score   : {response.get('confidence_score', 'N/A')}")
    print(f"Fallback applied   : {response.get('fallback_applied', False)}")

    if response.get("error"):
        print(f"\nValidation error: {response['error']}")
        return

    print("\n== AI Explanation ==")
    print(response.get("ai_explanation", ""))

    recommendations = response.get("recommendations", [])
    print(f"\n== Top {len(recommendations)} Recommendations ==")
    for i, product in enumerate(recommendations, 1):
        print(f"\n{i}. {product.get('product', '')} by {product.get('brand', '')}")
        print(
            "   Category : "
            f"{product.get('category', '')} > {product.get('sub_category', '')}"
        )
        print(
            "   Price    : "
            f"Rs. {product.get('sale_price', '')} | Rating: {product.get('rating', '')}"
        )

    if response.get("customer_signals"):
        print("\n== Customer Signals (from CSV) ==")
        for key, value in response["customer_signals"].items():
            if value not in ("", None):
                print(f"   {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Rahul")
    parser.add_argument("--age", type=int, default=29)
    parser.add_argument("--income", default="12 LPA")
    parser.add_argument("--interests", default="fitness,healthy food")
    parser.add_argument("--history", default="protein bars,almond milk,oats")
    args = parser.parse_args()

    persona = {
        "name": args.name,
        "age": args.age,
        "income": args.income,
        "interests": [x.strip() for x in args.interests.split(",") if x.strip()],
        "purchase_history": [x.strip() for x in args.history.split(",") if x.strip()],
    }
    run(persona)
