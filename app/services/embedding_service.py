"""
Embedding Service
-----------------
Upgraded from all-MiniLM-L6-v2 (384-dim, general) to
BAAI/bge-small-en-v1.5 (384-dim, retrieval-optimised).

BGE models are specifically fine-tuned for asymmetric retrieval tasks
(short query → long document), which matches our use-case exactly.
Same vector dimension (384) means NO Pinecone index rebuild needed.

Key changes vs v1:
- BGE requires a query prefix for retrieval tasks — applied automatically.
- generate_embeddings_batch() is now the primary path; single embed calls
  it internally so there is exactly one code path.
- show_progress_bar is suppressed by default so Streamlit logs stay clean.
"""

from sentence_transformers import SentenceTransformer

# BGE-small: same 384-dim as MiniLM but significantly better on MTEB retrieval
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE retrieval prefix — must be prepended to *query* text (not documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BATCH_SIZE = 256  # bigger batches = faster; fits in CPU RAM comfortably

_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the model (avoids reload on every Streamlit rerun)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embedding(text: str, is_query: bool = True) -> list[float]:
    """
    Generate a single embedding vector.

    Parameters
    ----------
    text     : input string
    is_query : True for persona/query text, False for product documents.
               BGE applies a prefix only to queries.
    """
    return generate_embeddings_batch([text], is_query=is_query)[0]


def generate_embeddings_batch(
    texts: list[str],
    is_query: bool = False,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Parameters
    ----------
    texts        : list of strings
    is_query     : True when embedding persona/search queries
    show_progress: True to display tqdm bar (useful for CLI ingestion)

    Returns
    -------
    List of float lists (one per input text), ready for Pinecone upsert.
    """
    model = _get_model()
    cleaned = [str(t).strip() or " " for t in texts]

    if is_query:
        cleaned = [BGE_QUERY_PREFIX + t for t in cleaned]

    vectors = model.encode(
        cleaned,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # cosine sim == dot product after normalisation
    )
    return [v.tolist() for v in vectors]
