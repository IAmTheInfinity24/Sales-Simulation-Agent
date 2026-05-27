# Sales AI Agent

An agentic product recommendation application built with Streamlit, LangGraph,
LangChain, Pinecone, Sentence Transformers, and a local Ollama LLM.

The app recommends products for a customer persona using interests, purchase
history, catalog embeddings, customer enrichment data, feedback memory, and
fallback routing when retrieval confidence is low.

## Features

- Streamlit web UI with product catalog upload and recommendation workflow
- Agentic recommendation graph built with LangGraph
- Pinecone vector database for semantic product retrieval
- BGE-small embedding model for query and product embeddings
- Local Ollama Llama 3 model for recommendation explanations
- Customer persona validation and context generation
- Intent detection for domains such as fitness, travel, riding, beauty, sexual wellness, and more
- Keyword plus semantic re-ranking for better product relevance
- Duplicate product filtering and category diversity handling
- Low-confidence fallback flow with broad search and category-popular fallback
- Feedback buttons that update a local user preference profile
- Memory-aware routing that can use preference-boosted retrieval after enough feedback
- CLI test harness for quick terminal checks

## Project Structure

```text
sales-ai-agent/
  app/
    agents/
      feedback_agent.py          # Saves like/dislike feedback and updates user memory
      persona_agent.py           # Builds and validates persona context
      recommendation_agent.py    # Retrieves, ranks, deduplicates, and explains products
      router_agent.py            # Chooses standard vs preference-boosted retrieval
    data/
      customers.csv              # Dummy customer enrichment dataset
      products.csv               # Dummy product catalog dataset
    graph/
      agentic_graph.py           # Current LangGraph agentic workflow
      sales_graph.py             # Older fixed workflow kept for compatibility
    memory/
      user_store.py              # Local JSON preference memory
    services/
      embedding_service.py       # BGE-small embedding generation
      llm_service.py             # Ollama/Llama explanation service
      pinecone_service.py        # Pinecone index, query, upsert, and stats helpers
    tools/
      catalog_sync.py            # Tracks local catalog sync status
    utils/
      customer_enrichment.py     # Looks up customer records from CSV
      intent_classifier.py       # Detects recommendation intents and scoring rules
      product_formatter.py       # Converts products into embedding text
    load_products.py             # CLI catalog ingestion into Pinecone
    main.py                      # CLI test harness
    streamlit_app.py             # Main web application
  .env.example                   # Example environment variables
  .gitignore
  requirements.txt
  README.md
```

## How The Application Works

1. The user uploads or indexes a product catalog.
2. Product rows are converted into natural-language product descriptions.
3. Product descriptions are embedded using `BAAI/bge-small-en-v1.5`.
4. Vectors are stored in Pinecone under the default namespace `__default__`.
5. The user enters a customer persona in the Streamlit UI.
6. The LangGraph workflow validates the persona and loads any stored user memory.
7. The router decides whether to use standard retrieval or preference-boosted retrieval.
8. The recommendation agent builds a semantic query from interests and purchase history.
9. Pinecone returns candidate products.
10. Candidates are re-ranked using semantic score, keyword score, customer enrichment, and intent rules.
11. The graph checks result confidence and applies fallback search when needed.
12. Ollama generates a concise explanation for the recommended products.
13. User feedback can be saved and used in future recommendations.

## Agentic Graph Flow

The current app uses `app/graph/agentic_graph.py`.

```text
persona_node
  -> validate_node
    -> error_node, if invalid
    -> memory_retrieve_node, if valid
      -> dynamic_router_node
        -> recommendation_node
          -> confidence_check_node
            -> fallback_node, if confidence is low
            -> diversity_check_node
              -> widen_query_node, if category diversity is too low
              -> END
```

Key routing behavior:

- `standard`: used for new users or users without enough feedback history.
- `preference_boosted`: used when a user has enough saved feedback and a preference embedding.
- `broad`: fallback search with a simplified persona and a larger candidate pool.
- `category_popular`: fallback search filtered toward preferred categories.

## Tech Stack

- Python 3.11+
- Streamlit
- LangGraph
- LangChain Community
- Pinecone
- Sentence Transformers
- BGE-small embedding model: `BAAI/bge-small-en-v1.5`
- Ollama with `llama3:latest`
- Pandas

## Environment Variables

Create a local `.env` file from `.env.example`.

```env
PINECONE_API_KEY=your_pinecone_key_here
```

Do not commit `.env`. It is ignored by `.gitignore`.

## Setup

Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama separately:

```bash
ollama pull llama3
ollama run llama3
```

The app currently uses:

```python
model="llama3:latest"
```

## Pinecone Setup

The Pinecone service automatically connects to or creates this index:

```text
Index name: sales-ai-agent
Dimension : 384
Metric    : cosine
Namespace : __default__
Cloud     : aws
Region    : us-east-1
```

The embedding dimension is 384 because `BAAI/bge-small-en-v1.5` produces
384-dimensional vectors.

## Load Products Into Pinecone

You can populate Pinecone in two ways.

### Option 1: Streamlit Upload

Run the app:

```bash
streamlit run app/streamlit_app.py
```

Then upload a product CSV from the left-side product upload panel.

Required columns:

```text
product, brand, category, sub_category, type, sale_price, market_price, rating, description
```

### Option 2: CLI Loader

Use the included dummy catalog:

```bash
python app/load_products.py
```

Or pass a custom CSV:

```bash
python app/load_products.py --csv path/to/products.csv
```

The loader deduplicates products by `product` and `brand`, generates embeddings
in batches, and upserts vectors into Pinecone.

## Run The Application

Start the Streamlit UI:

```bash
streamlit run app/streamlit_app.py
```

Run a quick terminal test:

```bash
python app/main.py
```

Run a custom terminal test:

```bash
python app/main.py --name "Aditya" --age 28 --income "12 LPA" --interests "riding,travelling" --history "bike"
```

## Feedback And Memory

The UI allows users to like or skip recommended products. When feedback is
saved, `feedback_agent.py` updates the local profile store under:

```text
app/data/user_profiles/
```

That folder is ignored by Git because it is runtime state.

After enough feedback exists, `router_agent.py` can route future requests to
`preference_boosted` retrieval. This blends the current query embedding with
the saved preference embedding so previous likes can influence future results.

## Data Files

This repo includes dummy datasets:

```text
app/data/products.csv
app/data/customers.csv
```

These are intentionally kept in the repo for demo and testing.

## Deployment Notes

The project can be pushed to GitHub after running:

```bash
git init
git add .
git status
git commit -m "Initial sales AI agent app"
```

Before pushing, confirm these are not staged:

```text
.env
.venv/
app/data/user_profiles/
app/data/sync_state.json
```

Important deployment limitation:

- The current app uses local Ollama.
- Streamlit Cloud and many serverless platforms do not run a local Ollama server.
- For cloud deployment, either deploy on a VM where Ollama is installed and running, or replace `llm_service.py` with an API-based LLM provider.

For Streamlit Cloud, set secrets in the deployment settings instead of using
`.env` directly.

## Common Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app/streamlit_app.py
```

Load catalog:

```bash
python app/load_products.py
```

Run CLI test:

```bash
python app/main.py
```

Check Python syntax:

```bash
python -m py_compile app/main.py app/streamlit_app.py app/load_products.py
```

## Known Limitations

- Recommendation quality depends heavily on the product catalog. If the catalog
  does not contain strong matches for a persona, fallback products may be broad.
- Existing Pinecone vectors should be regenerated if the embedding model or
  product text formatter changes.
- Ollama must be running locally before explanation generation can work.
- Some dependency combinations on Windows may require installing CPU-compatible
  PyTorch and torchvision builds.

## License

This project is currently for demo and learning use. Add a license before
publishing it as an open-source project.
