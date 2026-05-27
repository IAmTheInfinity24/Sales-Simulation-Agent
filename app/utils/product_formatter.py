"""
Product Formatter
-----------------
Builds the text string that gets embedded and stored in Pinecone.

Changes vs v1:
- Returns a coherent natural-language sentence rather than a raw key=value dump.
  This gives the embedding model much better signal because it was trained on prose.
- build_product_text() signature unchanged for backward compatibility.
- Added build_products_texts_batch() for fast batch ingestion.
"""


def build_product_text(row) -> str:
    """
    Convert a product row (dict or pandas Series) into a natural-language
    description suitable for embedding.

    Example output:
        "Protein Bar Chocolate Fudge by RiteBite Max Protein. Category: Food &
         Nutrition > Sports Nutrition. Price: ₹99. Rating: 4.3. Description:
         High-protein snack bar with 10g protein per bar."
    """
    product = str(row.get("product", "")).strip()
    brand = str(row.get("brand", "")).strip()
    category = str(row.get("category", "")).strip()
    sub_cat = str(row.get("sub_category", "")).strip()
    prod_type = str(row.get("type", "")).strip()
    price = str(row.get("sale_price", "")).strip()
    rating = str(row.get("rating", "")).strip()
    description = str(row.get("description", "")).strip()

    parts = []

    # Core identity
    if product and brand:
        parts.append(f"{product} by {brand}.")
    elif product:
        parts.append(f"{product}.")

    # Category hierarchy
    cat_parts = [c for c in [category, sub_cat, prod_type] if c]
    if cat_parts:
        parts.append(f"Category: {' > '.join(cat_parts)}.")

    # Price and rating
    if price:
        parts.append(f"Price: ₹{price}.")
    if rating:
        parts.append(f"Rating: {rating}.")

    # Description — most semantically rich part
    if description:
        # Truncate extremely long descriptions to avoid embedding noise
        parts.append(description[:400])

    return " ".join(parts)


def build_products_texts_batch(rows) -> list[str]:
    """
    Convert an iterable of product rows to a list of text strings.
    Designed for use with generate_embeddings_batch().
    """
    return [build_product_text(row) for row in rows]
