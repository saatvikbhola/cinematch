# 🔧 How Endee Is Used in CineMatch

> Complete reference of every Endee feature used in CineMatch, with full source code and explanations.

[Endee](https://github.com/endee-io/endee) is the **core data store and retrieval engine** powering CineMatch. Every movie is stored as a vector document inside Endee, and every user interaction — search, RAG, "find similar", taste matching — queries Endee to find semantically relevant results.

---

## Table of Contents

- [Endee Features Summary](#endee-features-summary)
- [1. Server Setup via Docker](#1-server-setup-via-docker)
- [2. Client Connection](#2-client-connection)
- [3. Index Creation with Hybrid Support](#3-index-creation-with-hybrid-support)
- [4. Document Ingestion (Batch Upsert)](#4-document-ingestion-batch-upsert)
- [5. Dense Vector Search](#5-dense-vector-search)
- [6. Sparse Vector Search (SPLADE)](#6-sparse-vector-search-splade)
- [7. Hybrid Queries (Dense + Sparse)](#7-hybrid-queries-dense--sparse)
- [8. Payload Filtering](#8-payload-filtering)
- [9. Metadata Storage & Retrieval](#9-metadata-storage--retrieval)
- [10. Document Count / Index Stats](#10-document-count--index-stats)
- [11. RAG Retrieval Layer](#11-rag-retrieval-layer)
- [12. "Find Similar" via Re-Embedding](#12-find-similar-via-re-embedding)

---

## Endee Features Summary

| # | Endee Feature | Where in CineMatch | Source File |
|---|---|---|---|
| 1 | Docker server deployment | Endee runs as a Docker container | `docker-compose.yml` |
| 2 | Python SDK client | Every module connects via `endee.Endee()` | `search.py`, `rag.py`, `ingest.py` |
| 3 | Index creation (dense + sparse) | Movies index with 384d dense + 30,522d sparse | `ingest.py` |
| 4 | Batch upsert with metadata + filters | ~5,000 movies indexed in memory-friendly chunks | `ingest.py` |
| 5 | Dense vector search | Semantic search by movie vibe/description | `search.py` |
| 6 | Sparse vector support | SPLADE sparse vectors for keyword precision | `embeddings.py`, `search.py` |
| 7 | Hybrid queries | Combined dense + sparse in single `index.query()` | `search.py`, `rag.py` |
| 8 | Payload filtering (`$eq`, `$range`) | Year, rating, language, status server-side filters | `search.py`, `rag.py` |
| 9 | Metadata payloads | 15+ fields returned with each result | `ingest.py`, `search.py` |
| 10 | Index stats (document count) | DB Explorer tab shows total indexed movies | `search.py` |
| 11 | RAG retrieval layer | Endee as the "R" in RAG | `rag.py` |
| 12 | Similar item lookup | "Find Similar" button re-queries Endee | `search.py` |

---

## 1. Server Setup via Docker

Endee runs as a standalone Docker container. CineMatch connects to it over HTTP.

```yaml
# docker-compose.yml — full file
services:
  endee:
    image: endeeio/endee-server:latest
    container_name: endee-server
    ports:
      - "8080:8080"
    ulimits:
      nofile: 100000
    logging:
      driver: "json-file"
      options:
        max-size: "200m"
        max-file: "5"
    environment:
      NDD_NUM_THREADS: 0
      NDD_AUTH_TOKEN: ""
    volumes:
      - endee-data:/data
    restart: unless-stopped

volumes:
  endee-data:
```

**Key configuration:**
- Port `8080` exposed for the Endee HTTP API
- Persistent volume `endee-data` ensures indexed data survives container restarts
- `NDD_NUM_THREADS: 0` uses all available CPU threads
- `ulimits.nofile: 100000` for high-concurrency workloads

---

## 2. Client Connection

Every module that interacts with Endee uses the same connection pattern via the Python SDK:

```python
# search.py — standard Endee client connection
from endee import Endee
from config import ENDEE_URL, ENDEE_INDEX_NAME

def _get_index():
    """Get the Endee movies index."""
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")         # http://localhost:8080/api/v1
    return client.get_index(name=ENDEE_INDEX_NAME)       # "movies_index"
```

```python
# rag.py — identical pattern for the RAG module
def _get_index():
    """Get the Endee movies index."""
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")
    return client.get_index(name=ENDEE_INDEX_NAME)
```

```python
# config.py — centralized Endee configuration
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_INDEX_NAME = "movies_index"
EMBEDDING_DIMENSION = 384
SPACE_TYPE = "cosine"
```

---

## 3. Index Creation with Hybrid Support

During ingestion, CineMatch creates an Endee index configured for **both** dense and sparse vector search:

```python
# ingest.py — creating the movies index
from endee import Endee, Precision
from config import ENDEE_URL, ENDEE_INDEX_NAME, EMBEDDING_DIMENSION, SPACE_TYPE

def create_endee_index(client: Endee):
    """Create the movies index in Endee (skip if exists)."""
    try:
        client.get_index(name=ENDEE_INDEX_NAME)
        print(f"Index '{ENDEE_INDEX_NAME}' already exists")
    except Exception:
        print(f"Creating index '{ENDEE_INDEX_NAME}' ({EMBEDDING_DIMENSION}d, {SPACE_TYPE})...")
        client.create_index(
            name=ENDEE_INDEX_NAME,          # "movies_index"
            dimension=EMBEDDING_DIMENSION,   # 384 (MiniLM embedding dimension)
            space_type=SPACE_TYPE,           # "cosine" similarity
            precision=Precision.FLOAT32,     # Full precision vectors
            sparse_dim=30522,                # SPLADE vocabulary size
        )
        print(f"Index created")
```

**Index configuration:**

| Parameter | Value | Why |
|---|---|---|
| `name` | `movies_index` | Descriptive index name |
| `dimension` | `384` | Matches `all-MiniLM-L6-v2` output dimension |
| `space_type` | `cosine` | Best for normalized sentence embeddings |
| `precision` | `FLOAT32` | Full precision for accurate similarity |
| `sparse_dim` | `30522` | SPLADE model vocabulary size (BERT tokenizer) |

---

## 4. Document Ingestion (Batch Upsert)

Movies are upserted into Endee in memory-friendly chunks. Each document has **four layers**: ID, dense vector, sparse vector, metadata payload, and filter fields.

```python
# ingest.py — the complete upsert logic for each movie
def index_movies(movies: list[dict], chunk_size: int = 50):
    """
    Embed movies and upsert into Endee in small memory-friendly chunks.
    Skips movies that are already indexed (tracked via indexed_ids.json).
    """
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")
    create_endee_index(client)
    index = client.get_index(name=ENDEE_INDEX_NAME)

    # Filter out already-indexed movies
    already_indexed = load_indexed_ids()
    new_movies = []
    for m in movies:
        if m["tmdb_id"] not in already_indexed:
            new_movies.append(m)

    if not new_movies:
        print(f"\n✅ All {len(movies)} movies are already indexed. Nothing to do!")
        return

    total = len(new_movies)
    newly_indexed = set()

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_movies = new_movies[chunk_start:chunk_end]

        # Step 1: Build text representations
        texts = []
        for m in chunk_movies:
            texts.append(build_movie_text(m))

        # Step 2: Dense embeddings (all-MiniLM-L6-v2)
        dense_vectors = embed_batch(texts, batch_size=16)

        # Step 3: Sparse embeddings (SPLADE)
        sparse_vectors = get_sparse_batch_embeddings(texts, batch_size=8)

        # Step 4: Build upsert batch and send to Endee
        batch = []
        for j, movie in enumerate(chunk_movies):
            sparse_indices, sparse_values = sparse_vectors[j]
            # Genres as list for proper filtering
            if isinstance(movie["genres"], list):
                genres_list = movie["genres"]
            else:
                genres_list = []
                for g in movie["genres"].split(","):
                    genres_list.append(g.strip())

            batch.append({
                "id": f"tmdb_{movie['tmdb_id']}",       # Unique document ID
                "vector": dense_vectors[j],               # 384-dim dense embedding
                "sparse_indices": sparse_indices,          # SPLADE token indices
                "sparse_values": sparse_values,            # SPLADE token weights
                "meta": {                                  # Metadata payload (returned with results)
                    "title": movie["title"],
                    "original_title": movie.get("original_title", ""),
                    "overview": movie["overview"],
                    "tagline": movie.get("tagline", ""),
                    "genres": ", ".join(genres_list),
                    "keywords": ", ".join(movie["keywords"][:10]) if isinstance(movie["keywords"], list) else movie["keywords"],
                    "cast": ", ".join(movie["cast"]) if isinstance(movie["cast"], list) else movie["cast"],
                    "director": movie["director"],
                    "year": movie["year"],
                    "rating": movie["rating"],
                    "vote_count": movie["vote_count"],
                    "runtime": movie["runtime"],
                    "language": movie["language"],
                    "spoken_languages": ", ".join(movie.get("spoken_languages", [])),
                    "production_companies": ", ".join(movie.get("production_companies", [])),
                    "production_countries": ", ".join(movie.get("production_countries", [])),
                    "budget": movie.get("budget", 0),
                    "revenue": movie.get("revenue", 0),
                    "status": movie.get("status", ""),
                    "collection": movie.get("belongs_to_collection", ""),
                    "poster_path": movie["poster_path"],
                    "backdrop_path": movie.get("backdrop_path", ""),
                    "tmdb_id": movie["tmdb_id"],
                    "popularity": movie.get("popularity", 0),
                },
                "filter": {                                # Server-side filterable fields
                    "genres": genres_list,
                    "language": movie["language"],
                    "production_companies": movie.get("production_companies", []),
                    "production_countries": movie.get("production_countries", []),
                    "status": movie.get("status", ""),
                    "year": movie["year"],
                    "rating": movie["rating"],
                    "vote_count": movie["vote_count"],
                    "runtime": movie["runtime"],
                    "popularity": movie.get("popularity", 0),
                },
            })

        # Batch upsert to Endee
        try:
            index.upsert(batch)
            for m in chunk_movies:
                newly_indexed.add(m["tmdb_id"])
        except Exception as e:
            print(f"Error upserting chunk: {e}")

        # Free memory aggressively
        del texts, dense_vectors, sparse_vectors, batch
        gc.collect()

    # Save updated indexed IDs for incremental runs
    save_indexed_ids(already_indexed | newly_indexed)
```

**Design decisions for ingestion:**
- **Chunked processing** (default 50 movies/chunk) — prevents OOM on low-RAM machines
- **Incremental indexing** via `indexed_ids.json` — re-running `ingest.py` only processes new movies
- **Aggressive memory cleanup** (`gc.collect()`) after each chunk
- **Separate `meta` vs `filter` fields** — `meta` stores display strings, `filter` stores structured data for Endee's filter engine

---

## 5. Dense Vector Search

The simplest form of search — embed a query and find nearest neighbors:

```python
# search.py — pure semantic search
def search_by_query(query: str, top_k: int = 20) -> list[dict]:
    """
    Semantic search — embed user query and find closest movies in Endee.

    Args:
        query: Natural language description (e.g., "mind-bending thriller with plot twists")
        top_k: Number of results to return

    Returns:
        List of movie dicts sorted by similarity
    """
    query_vector = embed_text(query)         # 384-dim dense embedding
    index = _get_index()
    results = index.query(                   # Endee dense vector query
        vector=query_vector, 
        top_k=top_k
    )
    return _format_results(results)
```

**What happens inside Endee:**
1. Receives the 384-dimensional query vector
2. Computes cosine similarity against all indexed movie vectors
3. Returns the `top_k` most similar documents with their metadata payloads

---

## 6. Sparse Vector Search (SPLADE)

CineMatch generates SPLADE sparse vectors for keyword-level precision. These are stored alongside dense vectors in Endee:

```python
# embeddings.py — SPLADE sparse embedding generation
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

_splade_model = None
_splade_tokenizer = None

def _get_splade() -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """Lazy-load the SPLADE model."""
    global _splade_model, _splade_tokenizer
    if _splade_model is None:
        model_id = "naver/splade-cocondenser-ensembledistil"
        _splade_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _splade_model = AutoModelForMaskedLM.from_pretrained(model_id)
        if torch.backends.mps.is_available():
             _splade_model = _splade_model.to("mps")
        elif torch.cuda.is_available():
             _splade_model = _splade_model.to("cuda")
        _splade_model.eval()
    return _splade_model, _splade_tokenizer

def get_sparse_embedding(text: str) -> tuple[list[int], list[float]]:
    """
    Generate sparse vectors using SPLADE.
    Returns: (sparse_indices, sparse_values)
    """
    model, tokenizer = _get_splade()
    
    device = next(model.parameters()).device
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    new_tokens = {}
    for k, v in tokens.items():
        new_tokens[k] = v.to(device)
    tokens = new_tokens
    
    with torch.no_grad():
        output = model(**tokens)
    
    # SPLADE pooling (max along the sequence dimension)
    vec = torch.max(
        torch.log(1 + torch.relu(output.logits)) * tokens["attention_mask"].unsqueeze(-1),
        dim=1
    )[0].squeeze()
    
    # Extract non-zero elements
    indices = vec.nonzero().squeeze(-1)
    values = vec[indices]
    
    return indices.cpu().tolist(), values.cpu().tolist()
```

**How SPLADE works with Endee:**
- SPLADE outputs a **sparse vector** over the BERT vocabulary (30,522 dimensions)
- Most values are zero — only relevant terms get non-zero weights
- Endee stores these as `sparse_indices` + `sparse_values` pairs
- At query time, Endee combines sparse similarity with dense similarity

---

## 7. Hybrid Queries (Dense + Sparse)

The main search function sends **both** dense and sparse vectors in a single Endee query call:

```python
# search.py — hybrid search query to Endee
results = index.query(
    vector=query_vector,                   # Dense: 384-dim semantic embedding
    sparse_indices=sparse_indices,         # Sparse: SPLADE token indices
    sparse_values=sparse_values,           # Sparse: SPLADE token weights
    top_k=fetch_k,                         # Number of results
    filter=filters if filters else None,   # Server-side metadata filters
)
```

**This is used in two places:**

1. **`search.py` → `search_with_filters()`** — Main search tab
2. **`rag.py` → `retrieve_from_endee()`** — RAG retrieval layer

```python
# rag.py — hybrid retrieval for RAG pipeline
def retrieve_from_endee(
    question: str,
    genres: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    language: str | None = None,
    production_companies: list[str] | None = None,
    status: str | None = None,
    taste_search_query: str = "",
    top_k: int = 8
) -> list[dict]:
    """Retrieve relevant movies from Endee using hybrid search."""
    search_text = question
    if taste_search_query:
        search_text = f"{question}. {taste_search_query}"

    query_vector = embed_text(search_text)
    sparse_indices, sparse_values = get_sparse_embedding(search_text)

    index = _get_index()
    
    filters = []
    # ... build filters ...
    
    fetch_k = max(100, top_k * 5)

    results = index.query(
        vector=query_vector,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        top_k=fetch_k,
        filter=filters if filters else None,
    )

    formatted = _format_results(results)
    # ... client-side filtering ...
    return formatted[:top_k]
```

**Why hybrid matters:** A query like *"Leonardo DiCaprio sci-fi"* needs both:
- **Dense vector** to match the sci-fi *vibe* (semantics)
- **Sparse vector** to match *"Leonardo DiCaprio"* exactly (keywords)

---

## 8. Payload Filtering

CineMatch uses Endee's server-side filter operators to narrow search results before ranking:

### `$eq` — Exact Match

Used for scalar fields like language and movie status:

```python
# search.py — exact match filters
filters = []
if language and language != "Any":
    filters.append({"language": {"$eq": language}})     # e.g., {"language": {"$eq": "ko"}}
if status and status != "Any":
    filters.append({"status": {"$eq": status}})          # e.g., {"status": {"$eq": "Released"}}
```

### `$range` — Numeric Range

Used for year and rating ranges:

```python
# search.py — range filters
if min_year and max_year:
    filters.append({"year": {"$range": [min_year, max_year]}})    # e.g., [2010, 2020]
elif min_year:
    filters.append({"year": {"$range": [min_year, 2026]}})
elif max_year:
    filters.append({"year": {"$range": [1900, max_year]}})

if min_rating:
    filters.append({"rating": {"$range": [min_rating, 10.0]}})    # e.g., [7.0, 10.0]
```

### RAG-Specific Default Filter

The RAG pipeline applies a quality gate — only movies rated 5.0+ are retrieved for answering questions:

```python
# rag.py — quality filter for RAG
actual_min_rating = min_rating if min_rating is not None else 5.0
filters.append({"rating": {"$range": [actual_min_rating, 10.0]}})
```

### Filter Composition

Multiple filters are passed as a list — Endee applies them as an AND condition:

```python
# Combined filter example
results = index.query(
    vector=query_vector,
    sparse_indices=sparse_indices,
    sparse_values=sparse_values,
    top_k=100,
    filter=[
        {"language": {"$eq": "en"}},                    # English only
        {"year": {"$range": [2015, 2025]}},             # Last 10 years
        {"rating": {"$range": [7.5, 10.0]}},            # Highly rated
        {"status": {"$eq": "Released"}},                 # Already released
    ],
)
```

### Graceful Fallback

If filters cause an error (e.g., unsupported format), CineMatch falls back to unfiltered search:

```python
# search.py — filter fallback
try:
    results = index.query(
        vector=query_vector, 
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        top_k=fetch_k, 
        filter=filters if filters else None,
    )
except Exception as e:
    print(f"⚠️ Filtered query failed ({e}), falling back to unfiltered")
    results = index.query(
        vector=query_vector, 
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        top_k=fetch_k
    )
```

---

## 9. Metadata Storage & Retrieval

Each movie document stores 15+ metadata fields in Endee's `meta` payload. These are returned with every search result:

```python
# search.py — extracting metadata from Endee results
def _format_results(raw_results, query_time: float = 0.0) -> list[dict]:
    """Format Endee query results into clean movie dicts."""
    movies = []
    
    for idx, r in enumerate(raw_results):
        meta = r.get("meta", {}) if isinstance(r, dict) else {}

        # Handle different result formats from Endee
        if hasattr(r, "meta"):
            meta = r.meta if isinstance(r.meta, dict) else {}
        if hasattr(r, "id"):
            doc_id = r.id
        else:
            doc_id = r.get("id", "")

        # Normalize score
        raw_score = getattr(r, "score", 0.0)
        normalized_sim = (raw_score + 1) / 2 if SPACE_TYPE == "cosine" else raw_score

        movie = {
            "id": doc_id,
            "similarity": round(normalized_sim, 4),
            "raw_similarity": raw_score,
            "title": meta.get("title", "Unknown"),
            "overview": meta.get("overview", ""),
            "tagline": meta.get("tagline", ""),
            "genres": meta.get("genres", ""),
            "keywords": meta.get("keywords", ""),
            "cast": meta.get("cast", ""),
            "director": meta.get("director", ""),
            "year": meta.get("year", 0),
            "rating": meta.get("rating", 0),
            "vote_count": meta.get("vote_count", 0),
            "runtime": meta.get("runtime", 0),
            "language": meta.get("language", "en"),
            "poster_url": get_poster_url(meta.get("poster_path", "")),
            "backdrop_url": get_backdrop_url(meta.get("backdrop_path", "")),
            "tmdb_id": meta.get("tmdb_id", 0),
            "production_companies": meta.get("production_companies", ""),
        }
        movies.append(movie)
    
    return movies
```

**Metadata fields stored in Endee:**

| Field | Type | Used For |
|---|---|---|
| `title` | string | Display, AI context |
| `original_title` | string | Non-English films |
| `overview` | string | Plot display, RAG context |
| `tagline` | string | UI display |
| `genres` | string | Display, client-side filter |
| `keywords` | string | RAG context |
| `cast` | string | Display, AI context |
| `director` | string | Display, AI context |
| `year` | int | Display, filter |
| `rating` | float | Display, filter |
| `vote_count` | int | Quality signal |
| `runtime` | int | Display |
| `language` | string | Filter |
| `production_companies` | string | Display, client-side filter |
| `poster_path` | string | Image display |
| `tmdb_id` | int | External linking |

---

## 10. Document Count / Index Stats

The Database Explorer tab displays real-time statistics from Endee:

```python
# search.py — getting index statistics
def get_db_stats() -> dict:
    """Get the number of documents currently stored in the Endee DB."""
    try:
        index = _get_index()
        return {
            "total_movies": index.count,     # Endee's built-in document count
        }
    except Exception as e:
        print(f"Error fetching DB stats: {e}")
        return {"total_movies": 0}
```

```python
# app.py — displaying stats in the UI
with tab_db:
    st.subheader("🗄️ Endee Database Metrics")
    
    with st.spinner("Fetching database statistics..."):
        stats = get_db_stats()
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Indexed Movies", f"{stats.get('total_movies', 0):,}")
    col2.metric("Vector Dimension", "384")
    col3.metric("Similarity Metric", "Cosine")
```

---

## 11. RAG Retrieval Layer

Endee serves as the "R" (Retrieval) in the RAG pipeline. The full flow:

1. **User asks a question** → e.g., *"What's a good movie for a first date?"*
2. **Hybrid retrieval from Endee** → top 8 movies with highest semantic + keyword similarity
3. **Context injection** → retrieved movies formatted as numbered references
4. **Gemini generation** → grounded answer citing specific movies by number

```python
# rag.py — full RAG pipeline
def rag_answer(api_key: str, question: str, ..., top_k: int = 8) -> dict:
    """Full RAG pipeline: Retrieve from Endee → Generate grounded answer with Gemini."""

    # Step 1: Retrieve relevant movies from Endee
    retrieved = retrieve_from_endee(question, ..., top_k=top_k)

    if not retrieved:
        return {
            "answer": "I couldn't find any relevant movies in the database.",
            "retrieved_movies": [],
            "num_retrieved": 0,
        }

    # Step 2: Build structured context from retrieved movies
    context = _build_context(retrieved)

    # Step 3: Generate grounded answer with Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=RAG_SYSTEM_PROMPT)

    prompt = f"""User Question: "{question}"

Retrieved Movies from Endee Vector Database (ranked by semantic similarity):

{context}

---

Using ONLY the retrieved movies above, answer the user's question.
Cite movies by their [number] reference (e.g., [1], [2]).
Explain WHY each recommended movie fits the question."""

    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "retrieved_movies": retrieved,
        "num_retrieved": len(retrieved),
    }
```

---

## 12. "Find Similar" via Re-Embedding

When a user clicks "Find Similar" on any movie card, CineMatch builds a text representation of that movie and queries Endee for nearest neighbors:

```python
# search.py — find movies similar to a given movie
def find_similar_by_text(movie_text: str, exclude_title: str = "", top_k: int = 10) -> list[dict]:
    """
    Find similar movies by embedding a movie's description text.

    Args:
        movie_text: Rich text description of source movie
        exclude_title: Title to exclude from results
        top_k: Number of results

    Returns:
        List of similar movies
    """
    query_vector = embed_text(movie_text)
    sparse_indices, sparse_values = get_sparse_embedding(movie_text)
    index = _get_index()
    t0 = time.perf_counter()
    results = index.query(
        vector=query_vector, 
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        top_k=top_k + 3              # Fetch extra to account for filtering out the source
    )
    query_time = time.perf_counter() - t0
    formatted = _format_results(results, query_time=query_time)

    # Exclude the source movie
    if exclude_title:
        formatted = [m for m in formatted if m["title"].lower() != exclude_title.lower()]

    return formatted[:top_k]
```

```python
# app.py — triggering "Find Similar" from the UI
if st.button("🔗 Find Similar", key=f"sim_{movie_idx}"):
    movie_text = f"{movie['title']}. {movie.get('overview', '')}. {movie.get('genres', '')}"
    with st.spinner("Finding similar movies..."):
        similar = find_similar_by_text(
            movie_text=movie_text,
            exclude_title=movie["title"],
            top_k=8,
        )
        if similar:
            st.session_state.search_results = similar
            st.session_state.ai_response = f"Showing movies similar to **{movie['title']}**"
            st.rerun()
```

---

## Endee API Methods Used — Quick Reference

| Method | Endee SDK Call | CineMatch Usage |
|---|---|---|
| Create index | `client.create_index(name, dimension, space_type, precision, sparse_dim)` | One-time setup during ingestion |
| Get index | `client.get_index(name)` | Every search, RAG, and stats call |
| Upsert | `index.upsert(batch)` | Batch insert during ingestion |
| Query (hybrid) | `index.query(vector, sparse_indices, sparse_values, top_k, filter)` | Every search and RAG retrieval |
| Count | `index.count` | DB Explorer stats display |
