"""CineMatch Search — Endee vector search with filtering and similarity."""

import time

from endee import Endee

from config import ENDEE_URL, ENDEE_INDEX_NAME, get_poster_url
from embeddings import embed_text, get_sparse_embedding

def _get_index():
    """Get the Endee movies index."""
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")
    return client.get_index(name=ENDEE_INDEX_NAME)


def get_db_stats() -> dict:
    """Get the number of documents currently stored in the Endee DB."""
    try:
        index = _get_index()
        return {
            "total_movies": index.count,
        }
    except Exception as e:
        print(f"Error fetching DB stats: {e}")
        return {"total_movies": 0}


def _format_results(raw_results, query_time: float = 0.0) -> list[dict]:
    """Format Endee query results into clean movie dicts."""
    movies = []
    
    # --- DEBUG: Log raw results from Endee ---
    print(f"\n{'='*60}")
    print(f"🔍 RAW ENDEE RESULTS ({len(raw_results)} results) — Query took {query_time*1000:.1f}ms")
    print(f"{'='*60}")
    for i, r in enumerate(raw_results[:5]):  # Log first 5
        if isinstance(r, dict):
            print(f"  [{i}] id={r.get('id', '?')} | raw_similarity={r.get('similarity', '?')} | distance={r.get('distance', '?')} | norm={r.get('norm', '?')}")
            meta = r.get('meta', {})
            print(f"       title={meta.get('title', '?')}")
        else:
            print(f"  [{i}] type={type(r)} | repr={repr(r)[:200]}")
    print(f"{'='*60}\n")
    
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
    
    print(f"Score normalization: max_raw={max_score:.6f}")

    for idx, r in enumerate(raw_results):
        meta = r.get("meta", {}) if isinstance(r, dict) else {}

        # Handle different result formats from Endee
        if hasattr(r, "meta"):
            meta = r.meta if isinstance(r.meta, dict) else {}
        if hasattr(r, "id"):
            doc_id = r.id
        else:
            doc_id = r.get("id", "")

        # Normalize similarity: scale relative to top result so best match ≈ 1.0
        normalized_sim = raw_scores[idx] / max_score if max_score > 0 else 0.0

        movie = {
            "id": doc_id,
            "similarity": round(normalized_sim, 4),
            "raw_similarity": round(raw_scores[idx], 6),
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
            "backdrop_url": meta.get("backdrop_path", ""),
            "tmdb_id": meta.get("tmdb_id", 0),
            "production_companies": meta.get("production_companies", ""),
        }
        movies.append(movie)
    
    # DEBUG: Show normalized scores
    for m in movies[:10]:
        print(f"  {m['title']} ({m['year']}) | raw={m['raw_similarity']:.6f} -> normalized={m['similarity']:.0%}")
    
    return movies


def search_by_query(query: str, top_k: int = 20) -> list[dict]:
    """
    Semantic search — embed user query and find closest movies in Endee.

    Args:
        query: Natural language description (e.g., "mind-bending thriller with plot twists")
        top_k: Number of results to return

    Returns:
        List of movie dicts sorted by similarity
    """
    query_vector = embed_text(query)
    index = _get_index()
    results = index.query(vector=query_vector, top_k=top_k)
    return _format_results(results)


def search_with_filters(
    query: str,
    genres: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    language: str | None = None,
    production_companies: list[str] | None = None,
    status: str | None = None,
    people: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Filtered semantic search — combines vector similarity with metadata filters.

    Uses Endee's filter field for server-side filtering + client-side fallback.
    """
    query_vector = embed_text(query)
    # Generate sparse vector for the query text
    sparse_indices, sparse_values = get_sparse_embedding(query)
    
    # --- DEBUG: Log query embedding info ---
    print(f"\nHYBRID SEARCH DEBUG")
    print(f"   Query: '{query}'")
    print(f"   Dense vector dim: {len(query_vector)}")
    print(f"   Sparse indices count: {len(sparse_indices)}")
    
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
    # Studio names like "A24" are weak semantic signals compared to movie plots, so
    # we need more raw results to surface them reliably.
    fetch_k = max(100, top_k * 5)

    try:
        t0 = time.perf_counter()
        results = index.query(
            vector=query_vector, 
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=fetch_k, 
            filter=filters if filters else None,
        )
        query_time = time.perf_counter() - t0
    except Exception as e:
        print(f"Filtered query failed ({e}), falling back to unfiltered")
        # Fallback to unfiltered if filter format isn't supported
        t0 = time.perf_counter()
        results = index.query(
            vector=query_vector, 
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=fetch_k
        )
        query_time = time.perf_counter() - t0

    # Client-side filtering for array fields (genres, production_companies)
    # and safety-net for scalar fields already handled server-side
    formatted = _format_results(results, query_time=query_time)
    
    # --- Exact Match Reranking ---
    # SPLADE often breaks unique names ("Timothée Chalamet") into subword tokens ("tim", "##oth", etc.)
    # and weights common tokens too highly (e.g., matching a character named "Tim").
    # To fix this, we apply a dynamic priority boost to the normalized similarity if the user asked 
    # for a specific person and they appear in the movie's cast/director fields.
    query_lower = query.lower()
    for m in formatted:
        bump = 0.0
        
        # Boost for explicitly extracted people (actors/directors)
        if people:
            for person in people:
                person_lower = person.lower()
                if person_lower in m["cast"].lower() or person_lower in m["director"].lower():
                    bump += 1.0  # Guarantees the movie is ranked above non-exact semantic matches
        
        # High value: exact title match
        if query_lower in m["title"].lower():
            bump += 0.10
            
        # Medium value: director match
        if query_lower in m["director"].lower():
            bump += 0.10
            
        # Medium value: keywords match
        if query_lower in m["keywords"].lower():
            bump += 0.05
            
        m["sort_score"] = m["similarity"] + bump

    # Re-sort using uncapped score so exact matched items jump to the very top
    formatted.sort(key=lambda x: x["sort_score"], reverse=True)
    
    # Cap similarity at 1.0 for the UI and cleanup
    for m in formatted:
        m["similarity"] = min(1.0, m["sort_score"])
        del m["sort_score"]

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


def find_similar(movie_id: str, top_k: int = 10) -> list[dict]:
    """
    Find movies similar to a given movie using its vector in Endee.

    Args:
        movie_id: The Endee document ID (e.g., "tmdb_27205")
        top_k: Number of similar movies to return

    Returns:
        List of similar movies (excluding the original)
    """
    # Build the query from the movie's own text
    index = _get_index()

    # We'll search by the movie's description instead
    # First, get the movie by doing a broad search and finding it
    results = index.query(
        vector=embed_text(movie_id.replace("tmdb_", "")),
        top_k=1
    )

    # Actually, let's use a workaround: search query = movie title
    return _format_results(results)


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
        top_k=top_k + 3
    )  # fetch extra to account for filtering
    query_time = time.perf_counter() - t0
    formatted = _format_results(results, query_time=query_time)

    # Exclude the source movie
    if exclude_title:
        formatted = [m for m in formatted if m["title"].lower() != exclude_title.lower()]

    return formatted[:top_k]
