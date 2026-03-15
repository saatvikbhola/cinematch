# 🚀 Setup & Execution Instructions

> Step-by-step guide to get CineMatch running — from zero to a fully-functional AI movie discovery engine.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Clone the Repository](#step-1-clone-the-repository)
- [Step 2: Start Endee Vector Database](#step-2-start-endee-vector-database)
- [Step 3: Configure Environment Variables](#step-3-configure-environment-variables)
- [Step 4: Install Python Dependencies](#step-4-install-python-dependencies)
- [Step 5: Ingest Movie Data into Endee](#step-5-ingest-movie-data-into-endee)
- [Step 6: Launch the Application](#step-6-launch-the-application)
- [Docker Deployment](#docker-deployment)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | How to Get |
|---|---|---|
| **Python** | 3.12+ | [python.org](https://www.python.org/downloads/) |
| **Docker** | Latest | [docker.com](https://docs.docker.com/get-docker/) |
| **Docker Compose** | v2+ | Included with Docker Desktop |
| **TMDb API Key** | Free | [developer.themoviedb.org](https://developer.themoviedb.org/docs/getting-started) |
| **Gemini API Key** | Free | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

> **Note:** The TMDb API key is required for data ingestion. The Gemini API key is optional but needed for AI features (explanations, RAG, taste analysis, query intent).

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/cinematch.git
cd cinematch
```

**Project structure after cloning:**

```
cinematch/
├── docker-compose.yml           # Endee server container
├── .gitignore
├── README.md
├── docs/
│   ├── SYSTEM_DESIGN.md
│   ├── ENDEE_INTEGRATION.md
│   └── SETUP.md                 # (this file)
└── cinematch/
    ├── app.py                   # Streamlit UI
    ├── search.py                # Endee search logic
    ├── rag.py                   # RAG pipeline
    ├── ai_chat.py               # Gemini AI chat
    ├── embeddings.py            # Dense + Sparse embeddings
    ├── ingest.py                # Data ingestion pipeline
    ├── taste_profile.py         # Letterboxd taste analysis
    ├── config.py                # Configuration
    ├── main.py                  # Entry point
    ├── requirements.txt         # Python dependencies
    └── pyproject.toml           # Project metadata
```

---

## Step 2: Start Endee Vector Database

Endee runs as a Docker container. From the project root:

```bash
docker compose up -d
```

This command:
- Pulls the `endeeio/endee-server:latest` Docker image
- Starts Endee on port **8080**
- Creates a persistent volume (`endee-data`) so your indexed data survives container restarts

**Verify Endee is running:**

```bash
curl http://localhost:8080/api/v1/health
```

Or simply check the container status:

```bash
docker ps
```

You should see `endee-server` running on port 8080.

**What `docker-compose.yml` does:**

```yaml
services:
  endee:
    image: endeeio/endee-server:latest
    container_name: endee-server
    ports:
      - "8080:8080"                # Expose Endee HTTP API
    ulimits:
      nofile: 100000               # High file descriptor limit for performance
    logging:
      driver: "json-file"
      options:
        max-size: "200m"
        max-file: "5"
    environment:
      NDD_NUM_THREADS: 0           # Use all available CPU threads
      NDD_AUTH_TOKEN: ""           # No authentication (local dev)
    volumes:
      - endee-data:/data           # Persistent storage
    restart: unless-stopped

volumes:
  endee-data:                      # Named volume for data persistence
```

---

## Step 3: Configure Environment Variables

Create a `.env` file inside the `cinematch/` directory:

```bash
# cinematch/.env
TMDB_API_KEY=your_tmdb_api_key_here
ENDEE_URL=http://localhost:8080
```

**Getting your TMDb API key:**
1. Go to [TMDb](https://www.themoviedb.org/) and create a free account
2. Navigate to Settings → API → Request a new API key
3. Copy the **API Key (v3 auth)** value

> **Security note:** The Gemini API key is **not** stored in `.env`. Instead, users enter it directly in the Streamlit sidebar at runtime. This ensures the key is never committed to version control or stored server-side.

**How config is loaded:**

```python
# config.py — loads environment variables from .env
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_READ_ACCESS_TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN", "")
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")

# TMDb
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
TMDB_POSTER_SIZE = "w500"
TMDB_BACKDROP_SIZE = "w1280"

# Endee
ENDEE_INDEX_NAME = "movies_index"
EMBEDDING_DIMENSION = 384
SPACE_TYPE = "cosine"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# Data
MOVIES_TO_FETCH = 5000
TMDB_RATE_LIMIT_DELAY = 0.26

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
```

---

## Step 4: Install Python Dependencies

```bash
cd cinematch
pip install -r requirements.txt
```

**Dependencies installed:**

| Package | Purpose |
|---|---|
| `endee` | Python SDK for Endee vector database |
| `streamlit` | Web UI framework |
| `sentence-transformers` | Dense embedding model (all-MiniLM-L6-v2) |
| `google-generativeai` | Gemini API client |
| `requests` | TMDb API calls |
| `python-dotenv` | Environment variable loading |
| `pandas` | Data table display |
| `tqdm` | Progress bars during ingestion |
| `Pillow` | Image handling |
| `tenacity` | Retry logic for API calls |

**With `uv` (alternative):**

```bash
cd cinematch
uv sync
```

> **First run note:** The first execution will download two ML models (~500MB total):
> - `all-MiniLM-L6-v2` (~90MB) — dense sentence embeddings
> - `naver/splade-cocondenser-ensembledistil` (~400MB) — sparse SPLADE embeddings

---

## Step 5: Ingest Movie Data into Endee

This is the critical step that populates Endee with movie vectors. The ingestion pipeline:

1. Fetches movie IDs from TMDb (multiple endpoints for diversity)
2. Fetches full details + credits + keywords for each movie (concurrent threads)
3. Builds rich text representations
4. Generates 384-dim dense embeddings (all-MiniLM-L6-v2)
5. Generates sparse embeddings (SPLADE)
6. Batch upserts into Endee with metadata + filter fields
7. Caches raw data locally for incremental re-runs

### Basic Ingestion

```bash
# Fetch and index 5,000 movies (default)
python ingest.py
```

### Quick Test (fewer movies)

```bash
# Fetch just 100 movies for a quick test
python ingest.py --count 100
```

### Custom Configuration

```bash
# Full control over the pipeline
python ingest.py --count 5000 --workers 8 --chunk-size 50
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--count` | `5000` | Number of movies to fetch from TMDb |
| `--workers` | `8` | Concurrent threads for TMDb API calls |
| `--chunk-size` | `50` | Movies per embedding batch (lower = less RAM) |
| `--skip-fetch` | off | Skip TMDb fetch, use cached data only |

### Re-index Cached Data

If you've already fetched movies and want to re-embed/re-index:

```bash
python ingest.py --skip-fetch
```

### Expected Output

```
============================================================
CineMatch Data Ingestion Pipeline
  Target: 5000 movies | Workers: 8 | Chunk size: 50
============================================================
Fetching movie IDs (target: 5000, skipping 0 existing, max 252 pages/endpoint)...
  Fetching from /movie/top_rated (up to 252 pages)...
    Page 1/252 | +20 new | Total: 20/5000 | 2.3 pages/s | ETA: 109s
    ...
Got 5000 unique movie IDs
Fetching details for 5000 movies using 8 threads...
Fetching movies: 100%|████████████████████| 5000/5000 [12:34<00:00, 6.63movies/s]
Fetched 4847 new movies in 754.2s

--- Chunk 1/97 (50 movies) ---
  Done in 8.2s | Progress: 50/4847 | ETA: 786s

--- Chunk 2/97 (50 movies) ---
  Done in 7.9s | Progress: 100/4847 | ETA: 743s
...

Successfully indexed 4847 new movies in 812.3s

============================================================
Ingestion complete in 1566.5s!
  Run 'streamlit run app.py' to start CineMatch.
============================================================
```

### Incremental Re-runs

The pipeline is **incremental** — running `ingest.py` again will only process new movies:

```
📊 4847 already indexed, 153 new to index (skipping duplicates)
```

This is tracked via `data_cache/indexed_ids.json`.

---

## Step 6: Launch the Application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### First-Time Setup in the App

1. Enter your **Gemini API Key** in the sidebar (unlocks AI features)
2. Start searching with natural language queries
3. Try the **RAG Q&A** tab for grounded movie answers
4. Explore the **Database Explorer** tab to see Endee statistics

### App Tabs

| Tab | Feature |
|---|---|
| 🔍 **Search & Discover** | Semantic search + AI explanations + follow-up chat |
| 🧠 **RAG Q&A** | Ask questions, get grounded answers with citations |
| 🗄️ **Database Explorer** | View Endee stats, browse indexed movies |

---

## Docker Deployment

For a fully containerized deployment:

### Option 1: Endee Only (Recommended for Development)

```bash
# Start Endee
docker compose up -d

# Run the Streamlit app locally
cd cinematch
pip install -r requirements.txt
python ingest.py --count 100
streamlit run app.py
```

### Option 2: Full Docker Build

```bash
# Start Endee
docker compose up -d

# Build and run the Streamlit app container (if Dockerfile is present)
cd cinematch
docker build -t cinematch .
docker run -p 7860:7860 cinematch
```

> **Note:** A Dockerfile is not included by default. You can create one for containerized deployment of the Streamlit app.

---

## Configuration Reference

All configuration is centralized in `cinematch/config.py`:

| Variable | Default | Description |
|---|---|---|
| `TMDB_API_KEY` | from `.env` | TMDb API key for data ingestion |
| `TMDB_READ_ACCESS_TOKEN` | from `.env` | TMDb v4 auth token (alternative) |
| `ENDEE_URL` | `http://localhost:8080` | Endee server address |
| `ENDEE_INDEX_NAME` | `movies_index` | Name of the Endee index |
| `EMBEDDING_DIMENSION` | `384` | Dense vector dimensions (must match model) |
| `SPACE_TYPE` | `cosine` | Similarity metric |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for AI features |
| `MOVIES_TO_FETCH` | `5000` | Default ingestion target |
| `TMDB_RATE_LIMIT_DELAY` | `0.26` | Delay between TMDb API calls (rate limiting) |

---

## Troubleshooting

### Endee container not starting

```bash
# Check container logs
docker logs endee-server

# Restart the container
docker compose down && docker compose up -d
```

### `ConnectionRefusedError` when running the app

Endee isn't running or isn't reachable. Verify:

```bash
curl http://localhost:8080/api/v1/health
```

If using a custom Endee URL, update `ENDEE_URL` in your `.env` file.

### Ingestion is slow

- Reduce `--chunk-size` to `25` if running out of RAM
- Increase `--workers` to `12` if your internet is fast
- Use `--skip-fetch` to re-index without re-fetching from TMDb

### Models downloading on first run

The first execution downloads ~500MB of ML models. This is a one-time cost:
- `all-MiniLM-L6-v2` → `~/.cache/torch/sentence_transformers/`
- SPLADE model → `~/.cache/huggingface/hub/`

### Missing Gemini API key

AI features (explanations, RAG, taste analysis) require a Gemini API key. Enter it in the sidebar of the running app. Without it, search still works — only AI-powered features are disabled.
