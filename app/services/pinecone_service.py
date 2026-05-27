"""
Pinecone Service
----------------
Changes vs v1:
- Namespace support added (pass namespace= to scope queries per catalog).
- get_pinecone_index() remains the shared accessor for backward compat.
- query_index() helper centralises all query calls with consistent defaults.
- Index dimension updated to 384 (unchanged — bge-small matches MiniLM dim).
- Added describe_stats() helper for the upload UI.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env")

pc = Pinecone(api_key=api_key)

INDEX_NAME = "sales-ai-agent"
VECTOR_DIM = 384
DEFAULT_NAMESPACE = "__default__"

# ---------------------------------------------------------------------------
# Create index if it doesn't exist
# ---------------------------------------------------------------------------
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created Pinecone index: {INDEX_NAME}")

_index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")


def get_pinecone_index():
    """Return the shared Pinecone index instance."""
    return _index


def query_index(
    vector: list[float],
    top_k: int = 60,
    namespace: str = DEFAULT_NAMESPACE,
    filter_dict: dict | None = None,
) -> dict:
    """
    Run a vector similarity query against the index.

    Parameters
    ----------
    vector     : embedding to search with
    top_k      : number of candidates to retrieve (default 60 for re-ranking)
    namespace  : Pinecone namespace (use different namespaces for catalogs)
    filter_dict: optional Pinecone metadata filter

    Returns
    -------
    Raw Pinecone query response dict.
    """
    kwargs = dict(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    if filter_dict:
        kwargs["filter"] = filter_dict

    return _index.query(**kwargs)


def upsert_vectors(vectors: list[dict], namespace: str = DEFAULT_NAMESPACE) -> None:
    """Upsert a list of vector dicts into the given namespace."""
    _index.upsert(vectors=vectors, namespace=namespace)


def describe_stats(namespace: str = DEFAULT_NAMESPACE) -> dict:
    """Return index stats filtered to the given namespace."""
    stats = _index.describe_index_stats()
    ns_stats = stats.get("namespaces", {}).get(namespace, {})
    return {
        "total_vector_count": ns_stats.get("vector_count", 0),
        "index_fullness": stats.get("index_fullness", 0),
        "dimension": stats.get("dimension", VECTOR_DIM),
    }
