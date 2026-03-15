"""CineMatch Configuration — loads environment variables and defines constants."""

import os
from dotenv import load_dotenv

# Load .env from the cinematch directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- API Keys ---
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_READ_ACCESS_TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN", "")
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")

# --- TMDb ---
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
TMDB_POSTER_SIZE = "w500"  # w92, w154, w185, w342, w500, w780, original
TMDB_BACKDROP_SIZE = "w1280"

# --- Endee ---
ENDEE_INDEX_NAME = "movies_index"
EMBEDDING_DIMENSION = 384
SPACE_TYPE = "cosine"

# --- Embedding Model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Gemini ---
GEMINI_MODEL = "gemini-2.5-flash"

# --- Data ---
MOVIES_TO_FETCH = 5000  # Total movies to pull from TMDb
TMDB_RATE_LIMIT_DELAY = 0.26  # ~4 requests/sec to stay under TMDb limit

def get_poster_url(poster_path: str) -> str:
    """Build full TMDb poster URL from path."""
    if not poster_path:
        return ""
    return f"{TMDB_IMAGE_BASE}/{TMDB_POSTER_SIZE}{poster_path}"

def get_backdrop_url(backdrop_path: str) -> str:
    """Build full TMDb backdrop URL from path."""
    if not backdrop_path:
        return ""
    return f"{TMDB_IMAGE_BASE}/{TMDB_BACKDROP_SIZE}{backdrop_path}"
