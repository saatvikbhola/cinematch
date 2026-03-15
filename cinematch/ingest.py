"""
CineMatch Data Ingestion — Fetches movies from TMDb, generates embeddings, and indexes in Endee.

Optimized for low-RAM machines: micro-batching, incremental ingestion, concurrent fetching.

Usage:
    python ingest.py                  # Fetch default 5000 movies
    python ingest.py --count 100      # Fetch 100 movies (quick test)
    python ingest.py --skip-fetch     # Re-embed and re-index cached data
    python ingest.py --workers 4      # Use 4 threads for TMDb API calls
"""

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from endee import Endee, Precision

from config import (
    TMDB_API_KEY,
    TMDB_BASE_URL,
    TMDB_RATE_LIMIT_DELAY,
    ENDEE_URL,
    ENDEE_INDEX_NAME,
    EMBEDDING_DIMENSION,
    SPACE_TYPE,
)
from embeddings import build_movie_text, embed_batch, get_sparse_batch_embeddings

# --- TMDb Fetching ---

HEADERS = {
    "accept": "application/json",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
INDEXED_IDS_FILE = os.path.join(CACHE_DIR, "indexed_ids.json")

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5)
)
def tmdb_get(endpoint: str, params: dict = None) -> dict:
    """Make a TMDb API request with rate limiting and exponential backoff retries."""
    if params is None:
        params = {}
    params["api_key"] = TMDB_API_KEY
    url = f"{TMDB_BASE_URL}{endpoint}"
    time.sleep(TMDB_RATE_LIMIT_DELAY)
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_movie_ids(count: int, skip_ids: set[int] | None = None) -> list[int]:
    """Fetch movie IDs from TMDb endpoints by paginating, skipping known IDs early."""
    movie_ids = set()
    skip_ids = skip_ids or set()
    endpoints = ["/movie/top_rated", "/discover/movie", "/movie/popular", "/movie/now_playing"]
    
    # TMDb allows up to 500 pages per endpoint, 20 results per page
    max_pages = min((count // 20) + 2, 500)

    print(f"Fetching movie IDs (target: {count}, skipping {len(skip_ids)} existing, max {max_pages} pages/endpoint)...")

    for endpoint in endpoints:
        if len(movie_ids) >= count:
            break
        print(f"  Fetching from {endpoint} (up to {max_pages} pages)...")
        endpoint_start = time.time()
        endpoint_new = 0
        for page in range(1, max_pages + 1):
            if len(movie_ids) >= count:
                print(f"    Reached target count ({count}), stopping early.")
                break
            try:
                data = tmdb_get(endpoint, {"page": page})
                results = data.get("results", [])
                
                # Faster duplicate detection
                page_ids = set()
                for m in results:
                    page_ids.add(m["id"])
                
                # 1. Skip IDs already in the database/cache
                page_ids -= skip_ids
                
                # 2. Skip IDs already found in this run
                new_ids = page_ids - movie_ids
                
                # 3. Add to final collection
                movie_ids.update(new_ids)
                
                new_this_page = len(new_ids)
                endpoint_new += new_this_page

                # Log every 10 pages or on the first page
                if page == 1 or page % 10 == 0:
                    elapsed = time.time() - endpoint_start
                    rate = page / max(elapsed, 0.1)
                    remaining_pages = max_pages - page
                    eta = remaining_pages / max(rate, 0.1)
                    print(f"    Page {page}/{max_pages} | +{new_this_page} new | Total: {len(movie_ids)}/{count} | {rate:.1f} pages/s | ETA: {eta:.0f}s")

                # Stop if TMDb returns no more results (end of data)
                total_pages = data.get("total_pages", max_pages)
                if page >= total_pages:
                    print(f"    Reached last available page ({total_pages}) for {endpoint}")
                    break
            except Exception as e:
                print(f"    Error on page {page} of {endpoint}: {e}")
                break

        endpoint_elapsed = time.time() - endpoint_start
        print(f"  done {endpoint}: +{endpoint_new} IDs in {endpoint_elapsed:.1f}s (total unique: {len(movie_ids)})")

    movie_ids_list = list(movie_ids)[:count]
    print(f"Got {len(movie_ids_list)} unique movie IDs")
    return movie_ids_list


def fetch_movie_details(movie_id: int) -> dict | None:
    """Fetch full movie details + credits + keywords from TMDb."""
    try:
        details = tmdb_get(f"/movie/{movie_id}", {
            "append_to_response": "keywords,credits"
        })

        genres = []
        for g in details.get("genres", []):
            genres.append(g["name"])
            
        kw_data = details.get("keywords", {})
        keywords = []
        for k in kw_data.get("keywords", []):
            keywords.append(k["name"])
            
        credits = details.get("credits", {})
        cast = []
        for c in credits.get("cast", [])[:5]:
            cast.append(c["name"])

        director = ""
        for crew_member in credits.get("crew", []):
            if crew_member.get("job") == "Director":
                director = crew_member["name"]
                break

        # Prepare the base movie dictionary
        result = {
            "tmdb_id": movie_id,
            "title": details.get("title", ""),
            "original_title": details.get("original_title", ""),
            "overview": details.get("overview", ""),
            "tagline": details.get("tagline", ""),
            "genres": genres,
            "keywords": keywords,
            "cast": cast,
            "director": director,
            "rating": round(details.get("vote_average", 0), 1),
            "vote_count": details.get("vote_count", 0),
            "runtime": details.get("runtime", 0) or 0,
            "language": details.get("original_language", "en"),
            "spoken_languages": [],
            "production_companies": [],
            "production_countries": [],
            "budget": details.get("budget", 0),
            "revenue": details.get("revenue", 0),
            "status": details.get("status", ""),
            "belongs_to_collection": details.get("belongs_to_collection", {}).get("name") if details.get("belongs_to_collection") else None,
            "poster_path": details.get("poster_path", ""),
            "backdrop_path": details.get("backdrop_path", ""),
            "popularity": details.get("popularity", 0),
        }

        # Year extraction
        release_date = details.get("release_date")
        if release_date:
            result["year"] = int(release_date[:4])
        else:
            result["year"] = 0
        
        # Populate lists with explicit loops
        for sl in details.get("spoken_languages", []):
            result["spoken_languages"].append(sl.get("name"))
            
        for pc in details.get("production_companies", []):
            result["production_companies"].append(pc.get("name"))
            
        for pc in details.get("production_countries", []):
            result["production_countries"].append(pc.get("name"))

        return result
    except Exception as e:
        print(f"Error fetching movie {movie_id}: {e}")
        return None


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
    print(f"Need to fetch {len(new_ids)} new movies ({len(cached_ids)} already cached, {len(indexed_ids)} already indexed)")

    if not new_ids:
        return cached[:count]

    # Concurrent fetching with ThreadPoolExecutor
    new_movies = []
    t0 = time.time()
    
    print(f"Fetching details for {len(new_ids)} movies using {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_movie_details, mid): mid for mid in new_ids}
        
        with tqdm(total=len(futures), desc="Fetching movies") as pbar:
            for future in as_completed(futures):
                movie = future.result()
                if movie and movie["overview"]:
                    new_movies.append(movie)
                pbar.update(1)
    
    elapsed = time.time() - t0
    print(f"Fetched {len(new_movies)} new movies in {elapsed:.1f}s ({len(new_movies)/max(elapsed,1):.1f} movies/sec)")

    # Merge with cache
    all_movies = cached + new_movies
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_movies, f, ensure_ascii=False, indent=2)

    print(f"Total cached: {len(all_movies)} movies")
    return all_movies[:count]


# --- Endee Indexing ---

def create_endee_index(client: Endee):
    """Create the movies index in Endee (skip if exists)."""
    try:
        client.get_index(name=ENDEE_INDEX_NAME)
        print(f"UPSERTING TO ENDEE INDEX: '{ENDEE_INDEX_NAME}' already exists")
    except Exception:
        print(f"Creating index '{ENDEE_INDEX_NAME}' ({EMBEDDING_DIMENSION}d, {SPACE_TYPE})...")
        client.create_index(
            name=ENDEE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            space_type=SPACE_TYPE,
            precision=Precision.FLOAT32,
            sparse_dim=30522,
        )
        print(f"Index created")


def load_indexed_ids() -> set:
    """Load the set of tmdb_ids already indexed in Endee."""
    if os.path.exists(INDEXED_IDS_FILE):
        with open(INDEXED_IDS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_indexed_ids(ids: set):
    """Persist the set of indexed tmdb_ids to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(INDEXED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f)


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
        print(f"\nAll {len(movies)} movies are already indexed. Nothing to do!")
        return

    print(f"\n{len(already_indexed)} already indexed, {len(new_movies)} new to index (skipping duplicates)")

    total = len(new_movies)
    total_start = time.time()
    
    print(f"Processing {total} movies in chunks of {chunk_size} (memory-friendly mode)")
    print(f"Each chunk: build text -> dense embed -> sparse embed -> upsert -> free RAM\n")

    newly_indexed = set()

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_movies = new_movies[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1
        total_chunks = (total + chunk_size - 1) // chunk_size
        
        print(f"--- Chunk {chunk_num}/{total_chunks} ({len(chunk_movies)} movies) ---")
        t0 = time.time()

        # Step 1: Build text representations
        texts = []
        for m in chunk_movies:
            texts.append(build_movie_text(m))

        # Step 2: Dense embeddings (small batch for CPU)
        dense_vectors = embed_batch(texts, batch_size=16)

        # Step 3: Sparse embeddings (small batch to prevent hanging)
        sparse_vectors = get_sparse_batch_embeddings(texts, batch_size=8)

        # Step 4: Build upsert batch and send to Endee
        batch = []
        for j, movie in enumerate(chunk_movies):
            sparse_indices, sparse_values = sparse_vectors[j]
            # Genres as list for proper $in filtering
            if isinstance(movie["genres"], list):
                genres_list = movie["genres"]
            else:
                genres_list = []
                for g in movie["genres"].split(","):
                    genres_list.append(g.strip())
            batch.append({
                "id": f"tmdb_{movie['tmdb_id']}",
                "vector": dense_vectors[j],
                "sparse_indices": sparse_indices,
                "sparse_values": sparse_values,
                "meta": {
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
                "filter": {
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

        try:
            index.upsert(batch)
            for m in chunk_movies:
                newly_indexed.add(m["tmdb_id"])
        except Exception as e:
            print(f"Error upserting chunk {chunk_num}: {e}")

        # Step 5: Free memory aggressively
        del texts, dense_vectors, sparse_vectors, batch
        gc.collect()

        elapsed = time.time() - t0
        done = chunk_end
        rate = done / (time.time() - total_start)
        eta = (total - done) / max(rate, 0.1)
        print(f"  Done in {elapsed:.1f}s | Progress: {done}/{total} | ETA: {eta:.0f}s\n")

    # Save updated indexed IDs
    save_indexed_ids(already_indexed | newly_indexed)

    total_elapsed = time.time() - total_start
    print(f"Successfully indexed {len(newly_indexed)} new movies in {total_elapsed:.1f}s ({len(newly_indexed)/max(total_elapsed,1):.1f} movies/sec)")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="CineMatch Data Ingestion")
    parser.add_argument("--count", type=int, default=5000, help="Number of movies to fetch")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip TMDb fetch, use cached data")
    parser.add_argument("--workers", type=int, default=8, help="Threads for TMDb fetching (default: 8)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Movies per embedding chunk (default: 50, lower = less RAM)")
    args = parser.parse_args()

    if not TMDB_API_KEY:
        print("TMDB_API_KEY not set. Check your .env file.")
        sys.exit(1)

    total_start = time.time()

    print("=" * 60)
    print("CineMatch Data Ingestion Pipeline")
    print(f"  Target: {args.count} movies | Workers: {args.workers} | Chunk size: {args.chunk_size}")
    print("=" * 60)

    if args.skip_fetch:
        cache_file = os.path.join(CACHE_DIR, "movies_raw.json")
        if not os.path.exists(cache_file):
            print("No cached data found. Run without --skip-fetch first.")
            sys.exit(1)
        with open(cache_file, "r", encoding="utf-8") as f:
            movies = json.load(f)
        movies = movies[:args.count]
        print(f"Loaded {len(movies)} movies from cache")
    else:
        movies = fetch_all_movies(args.count, max_workers=args.workers)

    index_movies(movies, chunk_size=args.chunk_size)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Ingestion complete in {total_elapsed:.1f}s!")
    print(f"  Run 'streamlit run app.py' to start CineMatch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
