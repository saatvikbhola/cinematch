# 🏗️ System Design & Technical Approach

> Deep dive into CineMatch's architecture, data pipelines, and the design decisions behind the AI-powered movie discovery engine.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Data Ingestion Pipeline](#data-ingestion-pipeline)
- [Embedding Strategy](#embedding-strategy)
- [Hybrid Search Pipeline](#hybrid-search-pipeline)
- [AI Query Intent Analysis](#ai-query-intent-analysis)
- [RAG (Retrieval-Augmented Generation) Pipeline](#rag-retrieval-augmented-generation-pipeline)
- [Taste Profiling Pipeline](#taste-profiling-pipeline)
- [Filter Architecture](#filter-architecture)
- [Module Responsibilities](#module-responsibilities)

---

## High-Level Architecture

CineMatch follows a **retrieval-augmented** architecture where Endee serves as the central vector store and the primary retrieval engine. Every user-facing feature — search, RAG Q&A, taste matching, "find similar" — ultimately queries Endee.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User (Browser)                              │
│                       Streamlit Frontend                            │
│              ┌──────────┬─────────────┬────────────┐                │
│              │  Search  │   RAG Q&A   │  DB Stats  │                │
│              │  & Chat  │             │  & Browse  │                │
│              └────┬─────┴──────┬──────┴─────┬──────┘                │
└───────────────────┼────────────┼────────────┼───────────────────────┘
                    │            │            │
┌───────────────────▼────────────▼────────────▼───────────────────────┐
│                     CineMatch Backend (Python)                      │
│                                                                      │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐      │
│  │  AI Chat     │  │  RAG Pipeline │  │  Query Intent        │      │
│  │  (ai_chat.py)│  │  (rag.py)     │  │  Analysis            │      │
│  │              │  │               │  │  (ai_chat.py)        │      │
│  └──────┬───────┘  └───────┬───────┘  └───────────┬──────────┘      │
│         │                  │                      │                  │
│         │    ┌─────────────▼──────────────────────▼─────────┐       │
│         │    │           Search Module (search.py)           │       │
│         │    │  • Builds Endee queries with filters          │       │
│         │    │  • Normalizes results                         │       │
│         │    │  • Client-side array field filtering           │       │
│         │    └─────────────┬─────────────────────────────────┘       │
│         │                  │                                         │
│  ┌──────▼──────────────────▼─────────────────────────────────┐      │
│  │              Embeddings Module (embeddings.py)             │      │
│  │   Dense:  SentenceTransformer — all-MiniLM-L6-v2 (384d)  │      │
│  │   Sparse: SPLADE — naver/splade-cocondenser-ensembledistil│      │
│  └─────────────────────────┬─────────────────────────────────┘      │
│                            │                                         │
│  ┌─────────────────────────▼─────────────────────────────────┐      │
│  │           Data Ingestion (ingest.py)                       │      │
│  │   TMDb API → Parse → Build Text → Embed → Upsert          │      │
│  └─────────────────────────┬─────────────────────────────────┘      │
│                            │                                         │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    Endee Vector Database     │
              │                              │
              │  • Dense vectors (384d)      │
              │  • Sparse vectors (30,522d)  │
              │  • Metadata payloads         │
              │  • Filter fields             │
              │  • Cosine similarity          │
              └─────────────────────────────┘
```

---

## Data Ingestion Pipeline

The ingestion pipeline (`ingest.py`) is the first stage — it populates Endee with movie data. The design optimizes for low-RAM machines through micro-batching and incremental indexing.

### Pipeline Stages

```
TMDb API ──→ Fetch IDs ──→ Fetch Details ──→ Build Rich Text ──→ Dense Embed ──→ Sparse Embed ──→ Upsert to Endee
  (1)          (2)            (3)              (4)                 (5)             (6)              (7)
```

### Stage 1–3: Data Acquisition

Movie IDs are fetched from multiple TMDb endpoints (`/movie/top_rated`, `/discover/movie`, `/movie/popular`, `/movie/now_playing`) using concurrent threads for speed. Each movie's full details, credits, and keywords are then fetched individually.

```python
# ingest.py — concurrent fetching with ThreadPoolExecutor
def fetch_all_movies(count: int, max_workers: int = 8) -> list[dict]:
    """Fetch all movie data using concurrent threads. Uses cache for already-fetched movies."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "movies_raw.json")

    # Load existing cache
    cached = []
    cached_ids = set()
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        
        cached_ids = set()
        for m in cached:
            cached_ids.add(m["tmdb_id"])
            
        if len(cached) >= count:
            print(f"Using cached data ({len(cached)} movies)")
            return cached[:count]
        print(f"Cache has {len(cached)} movies, need {count}. Fetching more...")

    # Fetch IDs and filter out already-cached or already-indexed ones
    indexed_ids = load_indexed_ids()
    skip_set = cached_ids | indexed_ids
    
    movie_ids = fetch_movie_ids(count, skip_ids=skip_set)
    
    new_ids = []
    for mid in movie_ids:
        if mid not in cached_ids:
            new_ids.append(mid)

    # Concurrent fetching with ThreadPoolExecutor
    new_movies = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_movie_details, mid): mid for mid in new_ids}
        
        with tqdm(total=len(futures), desc="Fetching movies") as pbar:
            for future in as_completed(futures):
                movie = future.result()
                if movie and movie["overview"]:
                    new_movies.append(movie)
                pbar.update(1)

    # Merge with cache
    all_movies = cached + new_movies
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_movies, f, ensure_ascii=False, indent=2)

    return all_movies[:count]
```

**Design decisions:**
- **Multi-endpoint fetching** ensures genre diversity (popular + top-rated + now-playing + discover)
- **Local JSON cache** avoids re-fetching on subsequent runs
- **`indexed_ids.json` tracking** enables incremental indexing — only new movies are processed
- **Exponential backoff retry** (`tenacity`) handles TMDb rate limits gracefully

### Stage 4: Rich Text Construction

Each movie is transformed into a semantically rich text string that captures all searchable dimensions:

```python
# embeddings.py — building the text representation for embedding
def build_movie_text(movie: dict) -> str:
    """
    Build a rich text representation of a movie for embedding.
    Concatenates title, overview, genres, keywords, cast, director,
    language, and year for maximum semantic richness.
    """
    parts = []

    if movie.get("title"):
        parts.append(movie["title"])

    if movie.get("original_title") and movie["original_title"] != movie.get("title"):
        parts.append(f"Original title: {movie['original_title']}")

    if movie.get("belongs_to_collection"):
        parts.append(f"Part of the {movie['belongs_to_collection']} collection")

    if movie.get("tagline"):
        parts.append(movie["tagline"])

    if movie.get("overview"):
        parts.append(movie["overview"])

    if movie.get("genres"):
        genres = movie["genres"] if isinstance(movie["genres"], str) else ", ".join(movie["genres"])
        parts.append(f"Genres: {genres}")

    if movie.get("keywords"):
        keywords = movie["keywords"] if isinstance(movie["keywords"], str) else ", ".join(movie["keywords"])
        parts.append(f"Keywords: {keywords}")

    if movie.get("cast"):
        cast = movie["cast"] if isinstance(movie["cast"], str) else ", ".join(movie["cast"])
        parts.append(f"Starring: {cast}")

    if movie.get("director"):
        parts.append(f"Directed by: {movie['director']}")

    if movie.get("production_companies"):
        companies = ", ".join(movie["production_companies"]) if isinstance(movie["production_companies"], list) else movie["production_companies"]
        parts.append(f"Produced by: {companies}")

    if movie.get("production_countries"):
        countries = ", ".join(movie["production_countries"]) if isinstance(movie["production_countries"], list) else movie["production_countries"]
        parts.append(f"Production countries: {countries}")

    # Language name for semantic matching (e.g. "Hindi" not just "hi")
    lang_code = movie.get("language", "")
    lang_name = LANGUAGE_MAP.get(lang_code, lang_code.upper() if lang_code else "")
    if lang_name:
        parts.append(f"Primary language: {lang_name}")

    if movie.get("spoken_languages"):
        sl = ", ".join(movie["spoken_languages"]) if isinstance(movie["spoken_languages"], list) else movie["spoken_languages"]
        parts.append(f"Spoken languages: {sl}")

    if movie.get("year") and movie["year"] > 0:
        parts.append(f"Year: {movie['year']}")

    return ". ".join(parts)
```

**Why this approach?** The embedding model (`all-MiniLM-L6-v2`) is a general-purpose sentence transformer. By concatenating title, plot, genres, keywords, cast, director, language, and production info into a single string, we create a vector that encodes the movie's full identity — so a query like *"Christopher Nolan sci-fi"* naturally matches movies by that director in that genre.

### Stage 5–6: Dual Embedding Generation

Every movie gets **two** vector representations:

| Type | Model | Dimension | Purpose |
|---|---|---|---|
| **Dense** | `all-MiniLM-L6-v2` | 384 | Semantic meaning — captures mood, themes, plot similarity |
| **Sparse** | SPLADE (`naver/splade-cocondenser-ensembledistil`) | 30,522 | Keyword precision — exact term matching (actor names, titles) |

```python
# embeddings.py — dense embedding
def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

# embeddings.py — sparse embedding (SPLADE)
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

**Why hybrid?** Dense embeddings excel at understanding *"a movie about dreams within dreams"* but may miss exact terms like *"Leonardo DiCaprio"*. SPLADE sparse vectors fill that gap by learning which vocabulary terms are important, giving keyword-level precision while still benefiting from neural training.

### Stage 7: Endee Upsert with Metadata + Filters

Each movie is upserted into Endee with three data layers:

```python
# ingest.py — building each Endee document
batch.append({
    "id": f"tmdb_{movie['tmdb_id']}",
    "vector": dense_vectors[j],              # 384-dim dense vector
    "sparse_indices": sparse_indices,         # SPLADE indices
    "sparse_values": sparse_values,           # SPLADE weights
    "meta": {                                  # Payload metadata (returned with results)
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
        "genres": genres_list,                 # List for $in matching
        "language": movie["language"],         # Scalar for $eq matching
        "production_companies": movie.get("production_companies", []),
        "production_countries": movie.get("production_countries", []),
        "status": movie.get("status", ""),
        "year": movie["year"],                 # Numeric for $range
        "rating": movie["rating"],             # Numeric for $range
        "vote_count": movie["vote_count"],
        "runtime": movie["runtime"],
        "popularity": movie.get("popularity", 0),
    },
})
```

**Key design decision:** Metadata (`meta`) and filters (`filter`) are separate in Endee. The `meta` payload stores human-readable strings (e.g., `"genres": "Action, Comedy"`) returned with search results. The `filter` field stores structured data (e.g., `"genres": ["Action", "Comedy"]`) used for server-side query filtering. This separation enables fast filtered retrieval without parsing strings at query time.

---

## Embedding Strategy

### Why Two Models?

| Scenario | Dense Only | Sparse Only | Hybrid (CineMatch) |
|---|---|---|---|
| *"A dark thriller about obsession"* | ✅ Great | ❌ Poor | ✅ Great |
| *"Leonardo DiCaprio movies"* | ⚠️ Weak | ✅ Great | ✅ Great |
| *"Pixar films like Coco"* | ⚠️ Moderate | ⚠️ Moderate | ✅ Great |
| *"Something feel-good and nostalgic"* | ✅ Great | ❌ Poor | ✅ Great |

The hybrid approach ensures that **both** semantic understanding and keyword precision contribute to every search.

### Batch Processing

For ingestion efficiency, both models support batched processing:

```python
# embeddings.py — batch dense embeddings
def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Generate embeddings for a batch of texts efficiently."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    result = []
    for e in embeddings:
        result.append(e.tolist())
    return result

# embeddings.py — batch sparse embeddings
def get_sparse_batch_embeddings(texts: list[str], batch_size: int = 32) -> list[tuple[list[int], list[float]]]:
    """Generate sparse vectors for a batch of texts."""
    model, tokenizer = _get_splade()
    device = next(model.parameters()).device
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        new_tokens = {}
        for k, v in tokens.items():
            new_tokens[k] = v.to(device)
        tokens = new_tokens
        
        with torch.no_grad():
            output = model(**tokens)
            
        vecs = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens["attention_mask"].unsqueeze(-1),
            dim=1
        )[0]
        
        for vec in vecs:
            indices = vec.nonzero().squeeze(-1)
            values = vec[indices]
            results.append((indices.cpu().tolist(), values.cpu().tolist()))
            
    return results
```

---

## Hybrid Search Pipeline

When a user searches, the full pipeline is:

```
User Query
    │
    ▼
┌───────────────────────────┐
│  AI Intent Analysis       │  ← Gemini extracts hidden filters
│  (analyze_query_intent)   │     from natural language
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Merge Filters            │  ← AI filters + UI sidebar filters
│  (genres, year, rating,   │
│   language, studio)       │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Dual Embedding           │  ← Dense (MiniLM) + Sparse (SPLADE)
│  of refined query text    │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Endee Hybrid Query       │  ← Combined dense + sparse + filters
│  (over-fetch 100 results) │     Server-side: $eq, $range
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Client-Side Refinement   │  ← Genre substring, production company
│  (array fields Endee      │     matching, then trim to top_k
│   can't filter natively)  │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  AI Explanation           │  ← Gemini explains why each result
│  (explain_recommendations)│     matches the user's query
└───────────────────────────┘
```

### The Core Search Function

```python
# search.py — the main filtered search function
def search_with_filters(
    query: str,
    genres: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    language: str | None = None,
    production_companies: list[str] | None = None,
    status: str | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Filtered semantic search — combines vector similarity with metadata filters.
    Uses Endee's filter field for server-side filtering + client-side fallback.
    """
    query_vector = embed_text(query)
    # Generate sparse vector for the query text
    sparse_indices, sparse_values = get_sparse_embedding(query)

    index = _get_index()

    # Build filter list for Endee
    # NOTE: Endee's $in does NOT work on array-type filter fields (genres,
    #       production_companies) — it only matches scalars.  These are
    #       filtered client-side below instead.
    filters = []
    if language and language != "Any":
        filters.append({"language": {"$eq": language}})
    if status and status != "Any":
        filters.append({"status": {"$eq": status}})

    # Server-side $range filters for year and rating
    if min_year and max_year:
        filters.append({"year": {"$range": [min_year, max_year]}})
    elif min_year:
        filters.append({"year": {"$range": [min_year, 2026]}})
    elif max_year:
        filters.append({"year": {"$range": [1900, max_year]}})

    if min_rating:
        filters.append({"rating": {"$range": [min_rating, 10.0]}})

    # Always over-fetch to ensure a deep semantic pool for client-side filtering
    fetch_k = max(100, top_k * 5)

    try:
        results = index.query(
            vector=query_vector, 
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=fetch_k, 
            filter=filters if filters else None,
        )
    except Exception as e:
        # Fallback to unfiltered if filter format isn't supported
        results = index.query(
            vector=query_vector, 
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=fetch_k
        )

    # Client-side filtering for array fields (genres, production_companies)
    formatted = _format_results(results)
    if genres:
        formatted = [m for m in formatted if any(g.lower() in m["genres"].lower() for g in genres)]
    if language and language != "Any":
        formatted = [m for m in formatted if m["language"] == language]
    if production_companies:
        formatted = [
            m for m in formatted
            if any(pc.lower() in m.get("production_companies", "").lower() for pc in production_companies)
        ]

    return formatted[:top_k]
```

### Similarity Score Normalization

Raw similarity scores from Endee are normalized relative to the top result so the best match always shows ~100%:

```python
# search.py — score normalization
def _format_results(raw_results, query_time: float = 0.0) -> list[dict]:
    """Format Endee query results into clean movie dicts."""
    # Extract raw similarity scores to find the max for normalization
    raw_scores = []
    for r in raw_results:
        sim = 0.0
        if hasattr(r, "similarity"):
            sim = r.similarity
        elif isinstance(r, dict):
            sim = r.get("similarity", r.get("score", 0.0))
        raw_scores.append(float(sim))
    
    max_score = max(raw_scores) if raw_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    for idx, r in enumerate(raw_results):
        # Normalize similarity: scale relative to top result so best match ≈ 1.0
        normalized_sim = raw_scores[idx] / max_score if max_score > 0 else 0.0
        # ... build movie dict with normalized_sim
```

---

## AI Query Intent Analysis

Before searching Endee, Gemini analyzes the user's natural language query to extract structured filters:

```python
# ai_chat.py — extracting filters from natural language queries
def analyze_query_intent(api_key: str, query: str) -> dict:
    """
    Use Gemini to understand user's search intent and extract filters.
    Returns:
        {
            "refined_query": "semantic search text",
            "genres": ["Action", "Thriller"],
            "min_year": 2000,
            "min_rating": 7.0,
            ...
        }
    """
    if not api_key:
        return {"refined_query": query}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""Analyze this movie search query and extract structured information.

Query: "{query}"

Return a JSON object with these fields (omit fields that aren't mentioned or implied):
- "refined_query": the semantic search text to use for vector similarity search. 
  CRITICAL RULES for refined_query:
  1. Keep it as CLOSE to the original query as possible. Your job is to PRESERVE the vibe, not rewrite it.
  2. Keep ALL mood/vibe/aesthetic words (e.g., 'cyberpunk', 'noir', 'dreamy', 'gritty', 'feel-good', 'dark')
  3. Keep ALL movie/director/actor references (e.g., 'like Back to the Future', 'Wes Anderson style')
  4. Keep ALL thematic descriptors (e.g., 'time travel', 'heist', 'coming of age', 'revenge')
  5. ONLY remove pure filter tokens: explicit year numbers, rating numbers, and language names when used as a filter
  6. If in doubt, KEEP the word. Over-preserving is always better than over-stripping.
- "genres": list of genres if mentioned (use TMDb genre names)
- "production_companies": list of studios if mentioned (e.g., A24, Pixar)
- "min_year": earliest year if mentioned
- "max_year": latest year if mentioned  
- "min_rating": minimum rating (on 10 scale) if mentioned
- "language": language code if mentioned (en, fr, ko, ja, etc.)

Return ONLY valid JSON, no markdown formatting or explanation."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]

        import json
        return json.loads(text)
    except Exception:
        return {"refined_query": query}
```

**Example:** The query *"highly rated Korean thrillers from the 2010s"* gets decomposed into:

```json
{
  "refined_query": "intense thrillers",
  "genres": ["Thriller"],
  "language": "ko",
  "min_year": 2010,
  "max_year": 2019,
  "min_rating": 7.0
}
```

The `refined_query` goes to Endee for semantic search, while the extracted fields become Endee filter parameters.

### Filter Merging Strategy

AI-extracted filters are merged with the sidebar UI filters. **UI always takes priority** if explicitly set:

```python
# app.py — merging AI + UI filters
# Genres: union of both
merged_genres = set(selected_genres) if selected_genres else set()
if "genres" in ai_filters and isinstance(ai_filters["genres"], list):
    merged_genres.update(ai_filters["genres"])

# Years and Ratings: UI overrides AI if explicitly set
final_min_year = min_year if min_year > 1900 else ai_filters.get("min_year")
final_max_year = max_year if max_year < 2026 else ai_filters.get("max_year")
final_min_rating = min_rating if min_rating > 0 else ai_filters.get("min_rating")

# Language: UI takes priority
final_lang = language_options.get(selected_lang)
if final_lang is None and "language" in ai_filters:
    final_lang = ai_filters["language"]
```

---

## RAG (Retrieval-Augmented Generation) Pipeline

The RAG tab implements a true retrieval → generation pipeline. Notably, the retrieval layer now accepts the **full set of search filters** (genres, year, rating, language, production companies, status) just like the main search — enabling filtered RAG queries:

```
User Question + Sidebar Filters
    │
    ▼
┌─────────────────────────────┐
│  Hybrid Retrieval           │  ← Same dual-embedding + Endee query
│  from Endee (top_k=8)      │     as search, with full filter support
│  with quality filter        │     + quality gate (min_rating ≥ 5.0)
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  Build Numbered Context     │  ← Format retrieved movies as
│  [1] Title (Year)           │     structured text with metadata
│      Rating, Genres, etc.   │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  Gemini Generation          │  ← System prompt enforces grounding:
│  with grounding rules       │     "ONLY answer based on retrieved
│                             │      movies, cite by [number]"
└─────────────────────────────┘
```

### RAG Retrieval with Full Filters

```python
# rag.py — retrieval function now accepts all filter parameters
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
    """Retrieve relevant movies from Endee using hybrid search, optionally personalized."""
    search_text = question
    if taste_search_query:
        search_text = f"{question}. {taste_search_query}"

    query_vector = embed_text(search_text)
    sparse_indices, sparse_values = get_sparse_embedding(search_text)

    index = _get_index()

    # Build filter list (same operators as search.py)
    filters = []
    if language and language != "Any":
        filters.append({"language": {"$eq": language}})
    if status and status != "Any":
        filters.append({"status": {"$eq": status}})
    if min_year and max_year:
        filters.append({"year": {"$range": [min_year, max_year]}})
    elif min_year:
        filters.append({"year": {"$range": [min_year, 2026]}})
    elif max_year:
        filters.append({"year": {"$range": [1900, max_year]}})

    # Default quality gate: only movies rated 5.0+ unless user overrides
    actual_min_rating = min_rating if min_rating is not None else 5.0
    filters.append({"rating": {"$range": [actual_min_rating, 10.0]}})

    fetch_k = max(100, top_k * 5)

    results = index.query(
        vector=query_vector,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        top_k=fetch_k,
        filter=filters if filters else None,
    )

    formatted = _format_results(results)

    # Client-side filtering for array fields
    if genres:
        formatted = [m for m in formatted if any(g.lower() in m["genres"].lower() for g in genres)]
    if production_companies:
        formatted = [
            m for m in formatted
            if any(pc.lower() in m.get("production_companies", "").lower() for pc in production_companies)
        ]

    return formatted[:top_k]
```

### RAG System Prompt

```python
# rag.py — system prompt enforcing grounded generation
RAG_SYSTEM_PROMPT = """You are CineMatch RAG, a movie expert that ONLY answers 
based on retrieved movie data. You must NEVER invent or hallucinate information.

Rules:
- Answer the user's question using ONLY the retrieved movies provided below.
- Cite specific movies by their [number] reference, e.g. [1], [2].
- DETECTION: If the user mentions a specific movie and it is NOT in the retrieved list,
  EXPLICITLY state that you don't have that specific movie in your database.
- HONESTY: If the retrieved movies are not a good fit for the user's request,
  state clearly that you couldn't find a strong match.
- PERSONALIZATION: If a user taste profile is provided, use it to personalize
  your tone and explain why specifically THESE movies fit THEIR taste.
- Be conversational, concise, and engaging.
- Format your response with bullet points for readability.
- Include relevant details like year, rating, director, and genres when citing.
"""
```

### Context Building

```python
# rag.py — formatting retrieved movies as LLM context
def _build_context(movies: list[dict]) -> str:
    """Build a numbered context block from retrieved movies for the LLM."""
    context_parts = []
    for i, m in enumerate(movies, 1):
        context_parts.append(
            f"[{i}] {m['title']} ({m.get('year', 'N/A')})\n"
            f"    Rating: ⭐ {m.get('rating', 'N/A')}/10\n"
            f"    Genres: {m.get('genres', 'N/A')}\n"
            f"    Director: {m.get('director', 'N/A')}\n"
            f"    Cast: {m.get('cast', 'N/A')}\n"
            f"    Studio: {m.get('production_companies', 'N/A')}\n"
            f"    Plot: {m.get('overview', 'No overview available.')}\n"
            f"    Similarity: {m.get('similarity', 0):.0%}"
        )
    return "\n\n".join(context_parts)
```

---

## Taste Profiling Pipeline

CineMatch can analyze a user's Letterboxd export to build a taste DNA and generate a personalized search query:

```
Letterboxd CSV Upload
    │
    ▼
┌─────────────────────────────┐
│  Parse CSVs                 │  ← Extract ratings, reviews
│  (ratings.csv, reviews.csv) │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  Build Taste Summary        │  ← Categorize: loved (4-5★),
│  (loved, liked, disliked)   │     liked (3-4★), disliked (<3★)
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  Gemini Taste Analysis      │  ← Generates 2-3 paragraph profile
│                             │     + SEARCH_QUERY for Endee
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  Search Endee               │  ← Auto-generated query used
│  with taste query           │     for personalized discovery
└─────────────────────────────┘
```

The taste profile is also injected into AI explanations and RAG answers for personalized recommendations.

---

## Filter Architecture

CineMatch uses a two-tier filtering strategy:

### Tier 1: Server-Side (Endee Native)

These filters run inside Endee during the vector search — results that don't match are never returned:

| Filter | Endee Operator | Field Type |
|---|---|---|
| Language | `$eq` | Scalar string |
| Movie Status | `$eq` | Scalar string |
| Year Range | `$range` | Numeric |
| Rating Range | `$range` | Numeric |

### Tier 2: Client-Side (Post-Retrieval)

Array/complex fields that Endee's `$in` doesn't support natively on array-type filter fields are filtered after retrieval:

| Filter | Method | Why Client-Side |
|---|---|---|
| Genres | Substring match on comma-separated string | Endee `$in` works on scalars, not arrays |
| Production Companies | Substring match | Same reason |

**Over-fetching strategy:** To ensure enough results survive client-side filtering, we always fetch `max(100, top_k * 5)` results from Endee and trim after filtering.

---

## Module Responsibilities

| Module | Lines | Responsibility |
|---|---|---|
| `app.py` | 465 | Streamlit UI — 3 tabs (Search, RAG, DB Explorer), session state, filter sidebar |
| `search.py` | 270 | Endee query builder, hybrid search, result formatting, score normalization, debug logging |
| `rag.py` | 244 | RAG pipeline — retrieve from Endee with full filter support, build context, Gemini generation |
| `ai_chat.py` | 210 | Gemini chat sessions, vibe-preserving query intent analysis, recommendation explanations |
| `embeddings.py` | 200 | Dense (MiniLM) + Sparse (SPLADE) embedding generation, batch processing |
| `ingest.py` | 480 | TMDb data fetching, concurrent processing, Endee index creation + upsert |
| `taste_profile.py` | 190 | Letterboxd CSV parsing (ratings, reviews, diary, watchlist), taste summary, Gemini profile generation |
| `config.py` | 46 | Centralized configuration — API URLs, model names, constants, TMDb URL helpers |
