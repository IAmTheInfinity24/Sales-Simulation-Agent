"""
load_products.py  (v2)
----------------------
Batch ingestion script — run once from CLI to populate Pinecone.

v1 issues fixed:
- Row-by-row embedding loop → single generate_embeddings_batch() call.
  For 38K products this is ~15-20x faster.
- description was missing from metadata in v1 → now included.
- market_price also stored (useful for discount signal in UI).
- Vectors uploaded to named namespace ("products") not the default namespace.
- Duplicate (product, brand) pairs are dropped before embedding.
- Progress reporting via tqdm.

Usage:
    python load_products.py                        # uses default CSV path
    python load_products.py --csv path/to/file.csv
"""

import argparse
import time
import pandas as pd
from tqdm import tqdm

from services.embedding_service import generate_embeddings_batch
from services.pinecone_service import upsert_vectors, describe_stats, DEFAULT_NAMESPACE
from utils.product_formatter import build_products_texts_batch

BATCH_SIZE = 256   # vectors per Pinecone upsert call
DEFAULT_CSV = "app/data/products.csv"


def load_products(csv_path: str = DEFAULT_CSV, namespace: str = DEFAULT_NAMESPACE) -> None:

    print(f"Loading products from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.fillna("")

    # Deduplicate on (product, brand) — same logic as Streamlit upload page
    before = len(df)
    df = df.drop_duplicates(subset=["product", "brand"])
    print(f"Rows after dedup: {len(df)} (removed {before - len(df)} duplicates)")

    rows = df.to_dict("records")
    total = len(rows)

    print(f"Generating embeddings for {total} products (batch size {BATCH_SIZE})...")
    start = time.time()

    # Build all product texts first
    texts = build_products_texts_batch(rows)

    # Embed in one pass — this is where v1 was slow (one call per row)
    all_embeddings = generate_embeddings_batch(texts, is_query=False, show_progress=True)

    print(f"Embeddings done in {round(time.time() - start, 1)}s. Upserting to Pinecone...")

    vectors = []
    upserted = 0

    for i, (row, embedding) in enumerate(tqdm(zip(rows, all_embeddings), total=total, desc="Upserting")):
        vectors.append({
            "id": str(row.get("index", i)),
            "values": embedding,
            "metadata": {
                "product":      str(row.get("product", "")),
                "category":     str(row.get("category", "")),
                "sub_category": str(row.get("sub_category", "")),
                "type":         str(row.get("type", "")),
                "brand":        str(row.get("brand", "")),
                "sale_price":   str(row.get("sale_price", "")),
                "market_price": str(row.get("market_price", "")),   # NEW
                "rating":       str(row.get("rating", "")),
                "description":  str(row.get("description", "")),    # was missing in v1
            },
        })

        if len(vectors) >= BATCH_SIZE:
            upsert_vectors(vectors, namespace=namespace)
            upserted += len(vectors)
            vectors = []

    if vectors:
        upsert_vectors(vectors, namespace=namespace)
        upserted += len(vectors)

    elapsed = round(time.time() - start, 1)
    stats = describe_stats(namespace=namespace)
    print(f"\n✅ Done — {upserted} vectors upserted to namespace '{namespace}' in {elapsed}s.")
    print(f"   Index now contains {stats['total_vector_count']} vectors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load product catalog into Pinecone.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to products CSV")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Pinecone namespace")
    args = parser.parse_args()
    load_products(csv_path=args.csv, namespace=args.namespace)
