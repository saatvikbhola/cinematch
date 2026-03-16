<p align="center">
  <img src="https://img.shields.io/badge/Endee-Vector%20DB-blueviolet?style=for-the-badge" alt="Endee" />
  <img src="https://img.shields.io/badge/Gemini-2.5%20Flash-orange?style=for-the-badge&logo=google" alt="Gemini" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python" />
</p>

# 🎬 CineMatch — AI Movie Discovery Engine

> **Discover movies by vibe, not just genre** — powered by [Endee](https://github.com/endee-io/endee) Vector Database + Google Gemini

CineMatch is an AI-powered movie discovery application that lets users search for movies using **natural language descriptions** instead of traditional keyword searches. It combines **semantic vector search**, **hybrid retrieval (dense + sparse)**, **RAG-based Q&A**, and **personalized taste profiling** — all built on the [Endee](https://endee.io) open-source vector database.

---

## 📖 Detailed Documentation

| Document | Description |
|---|---|
| **[System Design & Technical Approach](docs/SYSTEM_DESIGN.md)** | Full architecture, data pipelines, embedding strategy, filter design, module breakdown — all with source code references |
| **[How Endee Is Used](docs/ENDEE_INTEGRATION.md)** | Every Endee feature used in CineMatch (12 features) with complete source code and explanations |
| **[Setup & Execution Instructions](docs/SETUP.md)** | Step-by-step guide from zero to running app, CLI reference, Docker deployment, troubleshooting |

---

## 📌 Problem Statement

Traditional movie search relies on exact keyword matching — users must know specific titles, actor names, or genres. This fails when users want to express subjective preferences:

- *"A mind-bending thriller that keeps you guessing"*
- *"Something like Wes Anderson but darker"*
- *"A feel-good animated movie for a rainy day"*

**CineMatch solves this** by encoding movies as high-dimensional vectors that capture semantic meaning — plot, mood, themes, visual style — and using Endee's vector similarity search to find movies that *feel* right, not just match keywords.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Search by vibe/description using 384-dimensional sentence embeddings |
| ⚡ **Hybrid Search** | Dense (SentenceTransformer) + Sparse (SPLADE) retrieval for best of both worlds |
| 🧠 **RAG Q&A** | Ask natural-language questions, get grounded answers with movie citations |
| 🤖 **AI Query Understanding** | Gemini extracts hidden filters (genres, years, studios) from natural language |
| 🧬 **Taste Profiling** | Upload Letterboxd exports → Gemini analyzes your viewing DNA |
| 🎛️ **Rich Filtering** | Genres, year range, rating, language, production house, status |
| 🔗 **"Find Similar"** | Click any movie to discover semantically similar films |
| 💬 **AI Chat** | Refine results conversationally with Gemini |
| 🐳 **Dockerized** | One-command deployment with Docker Compose |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
│                     Streamlit Frontend                          │
│            ┌──────────┬─────────────┬────────────┐              │
│            │  Search  │   RAG Q&A   │  DB Stats  │              │
│            └────┬─────┴──────┬──────┴─────┬──────┘              │
└─────────────────┼────────────┼────────────┼─────────────────────┘
                  │            │            │
┌─────────────────▼────────────▼────────────▼─────────────────────┐
│                    CineMatch Backend (Python)                   │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐     │
│  │  AI Chat    │  │  RAG Pipeline│  │  Query Intent       │     │
│  │  (Gemini)   │  │  (Retrieve → │  │  Analysis (Gemini)  │     │
│  │             │  │   Generate)  │  │                     │     │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘     │
│         │                │                     │                │
│  ┌──────▼────────────────▼─────────────────────▼────────────┐   │
│  │              Embeddings Module                           │   │
│  │   Dense: SentenceTransformer (all-MiniLM-L6-v2, 384d)    │   │
│  │ Sparse: SPLADE (naver/splade-cocondenser-ensembledistil) │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │    Endee Vector Database     │
              │                              │
              │  Index: movies_index         │
              │  Dimensions: 384             │
              │  Metric: Cosine Similarity   │
              │  Sparse Dim: 30,522 (SPLADE) │
              │  Vectors: ~5,000 movies      │
              │                              │
              │  Features Used:              │
              │  • Dense vector search       │
              │  • Sparse vector search      │
              │  • Hybrid queries            │
              │  • Payload filtering ($eq,   │
              │    $range, $in)              │
              │  • Metadata storage          │
              └──────────────────────────────┘
```

### Data Flow

1. **Ingestion** → TMDb API → movie details fetched → rich text built → dual embedding (dense + sparse) → upserted into Endee with metadata + filter fields
2. **Search** → user query → Gemini extracts filters → dual embedding → Endee hybrid query with server-side filters → client-side refinement → results + AI explanation
3. **RAG Q&A** → question → hybrid retrieval from Endee → context injection → Gemini generates grounded answer with citations

---

## 🔧 How Endee Is Used

Endee is the **core data store and retrieval engine** powering CineMatch. Every movie is stored as a vector document inside Endee, and every user interaction queries Endee to find semantically relevant results.

### Index Creation

```python
# From ingest.py — creating the Endee index with hybrid search support
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

client.create_index(
    name="movies_index",
    dimension=384,              # Dense embedding dimension
    space_type="cosine",        # Cosine similarity metric
    precision=Precision.FLOAT32,
    sparse_dim=30522,           # SPLADE vocabulary size for sparse vectors
)
```

### Document Ingestion (Dual Embeddings)

Each movie is indexed with **both** dense and sparse vectors, plus structured metadata and filter fields:

```python
# Each movie document sent to Endee contains:
{
    "id": "tmdb_27205",
    "vector": [0.023, -0.041, ...],        # 384-dim dense embedding
    "sparse_indices": [1547, 2903, ...],    # SPLADE token indices
    "sparse_values": [0.82, 0.65, ...],     # SPLADE token weights

    "meta": {                                # Searchable metadata
        "title": "Inception",
        "overview": "A thief who steals corporate secrets...",
        "genres": "Action, Science Fiction, Adventure",
        "director": "Christopher Nolan",
        "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, ...",
        "year": 2010,
        "rating": 8.4,
        # ... 15+ metadata fields
    },

    "filter": {                              # Server-side filterable fields
        "language": "en",
        "status": "Released",
        "year_norm": 110,                     # Normalized: 2010 - 1900
        "rating": 8.4,
        "vote_count": 35000,
        "runtime": 148,
        "popularity": 82.5,
        "genre_action": "yes",                # Dynamic boolean flag
        "genre_science_fiction": "yes",
        "genre_adventure": "yes",
        "company_warner_bros._pictures": "yes",
        "company_legendary_entertainment": "yes",
    }
}
```

### Hybrid Search Queries

CineMatch uses Endee's **hybrid search** — combining dense vector similarity (semantic meaning) with sparse vector matching (keyword precision):

```python
# From search.py — hybrid query with server-side filters
results = index.query(
    vector=dense_embedding,            # Semantic similarity
    sparse_indices=splade_indices,     # Keyword precision (SPLADE)
    sparse_values=splade_values,
    top_k=100,                         # Over-fetch for client-side refinement
    filter=[                           # Server-side filters via Endee
        {"language": {"$eq": "en"}},
        {"year_norm": {"$range": [100, 125]}},  # 2000-2025 normalized
        {"rating": {"$range": [7.0, 10.0]}},
        {"genre_action": {"$eq": "yes"}},        # Dynamic genre flag
    ],
)
```

### RAG Retrieval Pipeline

The RAG module uses Endee as the retrieval layer — fetching relevant movies and injecting them as context for Gemini to generate grounded answers:

```python
# From rag.py — retrieve → generate
retrieved_movies = retrieve_from_endee(question, top_k=8)  # Hybrid search
context = build_context(retrieved_movies)                    # Format for LLM
answer = gemini.generate(f"Answer using ONLY: {context}")   # Grounded answer
```

---

## 📁 Project Structure

```
endee-project/
├── docker-compose.yml           # Endee server container
├── .gitignore
├── README.md
├── docs/
│   ├── SYSTEM_DESIGN.md         # Architecture & technical approach
│   ├── ENDEE_INTEGRATION.md     # How Endee is used (with full code)
│   └── SETUP.md                 # Setup & execution instructions
└── cinematch/
    ├── app.py                   # Streamlit UI — search, RAG, DB explorer
    ├── search.py                # Endee hybrid search + filtering logic
    ├── rag.py                   # RAG pipeline — retrieve from Endee → Gemini
    ├── ai_chat.py               # Gemini-powered chat, query intent analysis
    ├── embeddings.py            # Dense (MiniLM) + Sparse (SPLADE) embeddings
    ├── ingest.py                # TMDb fetch → embed → upsert into Endee
    ├── taste_profile.py         # Letterboxd taste analysis via Gemini
    ├── config.py                # Centralized configuration
    ├── main.py                  # Entry point
    ├── requirements.txt         # Python dependencies
    └── pyproject.toml           # Project metadata
```

### Module Breakdown

| Module | Purpose | Endee Integration |
|---|---|---|
| `ingest.py` | Fetches ~5,000 movies from TMDb, builds embeddings, indexes into Endee | Creates index, generates dual embeddings, batch upserts |
| `search.py` | Semantic + filtered search with debug logging | Hybrid queries with `$eq`, `$range` filters |
| `rag.py` | Retrieval-Augmented Generation Q&A with full filter support | Retrieves context from Endee with genre, year, rating, language, studio, and status filters |
| `embeddings.py` | Dense (MiniLM 384d) + Sparse (SPLADE) embedding generation | Provides vectors for Endee indexing and querying |
| `ai_chat.py` | Gemini chat + query intent extraction with vibe-preserving rules | Extracted filters passed to Endee queries |
| `taste_profile.py` | Letterboxd export → taste DNA → search query | Taste query searched in Endee |
| `app.py` | Streamlit UI with 3 tabs | Orchestrates all Endee interactions |
| `config.py` | Centralized configuration + TMDb URL helpers | Endee URL, index name, model constants |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) (open-source, handles 1B+ vectors/node) |
| **Dense Embeddings** | `all-MiniLM-L6-v2` via SentenceTransformers (384 dimensions) |
| **Sparse Embeddings** | SPLADE (`naver/splade-cocondenser-ensembledistil`) for keyword precision |
| **LLM** | Google Gemini 2.5 Flash (explanations, RAG, taste analysis, intent parsing) |
| **Frontend** | Streamlit |
| **Movie Data** | TMDb API (~5,000 movies with full metadata) |
| **Containerization** | Docker + Docker Compose |
| **Language** | Python 3.12+ |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- TMDb API key ([get one free](https://developer.themoviedb.org/docs/getting-started))
- Gemini API key ([get one free](https://aistudio.google.com/app/apikey))

### Step 1: Clone the Repository

```bash
git clone https://github.com/saatvikbhola/cinematch.git
cd cinematch
```

### Step 2: Start Endee

```bash
docker compose up -d
```

This pulls the official Endee Docker image and starts the server on port `8080`.

### Step 3: Configure Environment Variables

Create a `.env` file inside the `cinematch/` directory:

```bash
# cinematch/.env
TMDB_API_KEY=your_tmdb_api_key_here
ENDEE_URL=http://localhost:8080
```

> **Note:** The Gemini API key is entered directly in the app's sidebar for security — it is never stored on the server.

### Step 4: Install Dependencies

```bash
cd cinematch
uv sync
```

### Step 5: Ingest Movie Data

```bash
# Fetch 5,000 movies from TMDb, generate embeddings, and index into Endee
uv run python ingest.py

# Quick test with fewer movies
uv run python ingest.py --count 100

# Adjust concurrency and batch sizes
uv run python ingest.py --count 5000 --workers 8 --chunk-size 50
```

The ingestion pipeline:
1. Fetches movie IDs from TMDb (paginated, multi-endpoint)
2. Fetches full movie details + credits + keywords (concurrent threads)
3. Builds rich text representations for each movie
4. Generates 384-dim dense embeddings (all-MiniLM-L6-v2)
5. Generates sparse embeddings (SPLADE)
6. Batch upserts into Endee with metadata + filter fields
7. Caches raw data locally for incremental re-runs

### Step 6: Launch the App

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter your **Gemini API key** in the sidebar to unlock AI features.

---

## 📖 Usage Guide

### 🔍 Search & Discover

Type a natural language description of what you're looking for:

- *"A dark psychological thriller with unreliable narrators"*
- *"Feel-good animated movies by Pixar"*
- *"Korean crime dramas from the 2000s"*

CineMatch will:
1. Use **Gemini** to extract hidden filters (genres, years, studios) from your query
2. Generate **dense + sparse embeddings** for the semantic portion
3. Run a **hybrid search** on Endee with server-side metadata filters
4. Display results ranked by vector similarity with AI-generated explanations

### 🧠 RAG Q&A

Switch to the **RAG Q&A** tab and ask questions like:
- *"What's a good movie for a first date?"*
- *"I want to get into Korean cinema — where should I start?"*
- *"Recommend something like Interstellar but more emotional"*

The pipeline retrieves relevant movies from Endee and generates a grounded answer with `[numbered]` citations.

### 🟢 Letterboxd Taste Profiling

1. Export your data from [Letterboxd](https://letterboxd.com/settings/data/)
2. Upload `ratings.csv` (and optionally `reviews.csv`) in the sidebar
3. Click **"Analyze My Taste"** — Gemini analyzes your viewing DNA
4. Use the auto-generated search query to discover movies tailored to your palate

---

## ⚙️ Configuration

All configuration lives in `cinematch/config.py`:

| Variable | Default | Description |
|---|---|---|
| `ENDEE_URL` | `http://localhost:8080` | Endee server address |
| `ENDEE_INDEX_NAME` | `movies_index` | Endee index name |
| `EMBEDDING_DIMENSION` | `384` | Dense vector dimensions |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for AI features |
| `MOVIES_TO_FETCH` | `5000` | Default TMDb fetch count |

---

## 🐳 Docker Deployment

The project includes a `docker-compose.yml` for Endee and a `Dockerfile` for the Streamlit app:

```bash
# Start Endee server
docker compose up -d

# Build and run the Streamlit app (optional, for containerized deployment)
cd cinematch
docker build -t cinematch .
docker run -p 7860:7860 cinematch
```

---

## 🧪 Endee Features Demonstrated

This project showcases multiple Endee capabilities in a real-world application:

| Endee Feature | How CineMatch Uses It |
|---|---|
| **Index Management** | Create, configure, and query the `movies_index` |
| **Dense Vector Search** | 384-dim cosine similarity for semantic movie matching |
| **Sparse Vector Search** | SPLADE 30,522-dim sparse vectors for keyword precision |
| **Hybrid Queries** | Combined dense + sparse in single query calls |
| **Payload Filtering** | `$eq` (language, status, genre/company flags), `$range` (year_norm, rating) server-side filters |
| **Metadata Storage** | 15+ metadata fields stored per document (title, cast, director, etc.) |
| **Batch Upsert** | Memory-efficient chunked indexing of 5,000+ movies |
| **Document Count** | Real-time database statistics displayed in UI |

---

## fixes
 - [16-03-2026]
   - Client side filtering for genres and production companies removed
       - bad idea resulted in result starvation.
       - for example searching for superhero movies with filters in the ui selected as horror.
       - endee fetches movies that are similar to the query passed to it.
       - genre filter were applied client side after fetching.
       - which resulted in sometimes no result, even if there were results.
       - the DB was doing what is was supposed to but the use of filters were flawed.
         
   - filtering genres and production companies server side instead
       - changed how the data was sent to db instead of sending these as arrays.
       - what we are doing now is flatening them into its own individual scalar fields so that the '$eq' operator can be used server side
       - now looking for a horror film is ``{"genre_action": {"$eq": "yes"}}``
       - there are more fields for the movie but by doing this there is no more result starvation and also we are not filtering them again after fetching the result from the DB.
         
   - Boosting result when specific names are asked in the query [this is done client side]
       - when searching for specfic actors name like "drama with timothee chalamet"
       - the tokenizer was seperating "timothee" into "tim","othee". whihc resulted in the cast that have tim in them to appear at the top while "timothee chalamet" was below
       - to fix these used ranking to boost the result  
  

---

## 📚 References

- [Endee — Open Source Vector Database](https://github.com/endee-io/endee)
- [Endee Documentation](https://docs.endee.io/quick-start)
- [Endee Python SDK](https://pypi.org/project/endee/)
- [SentenceTransformers — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [SPLADE — Sparse Lexical and Expansion Model](https://huggingface.co/naver/splade-cocondenser-ensembledistil)
- [Google Gemini API](https://ai.google.dev/)
- [TMDb API](https://developer.themoviedb.org/)
- [Streamlit](https://streamlit.io/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using <a href="https://endee.io">Endee Vector Database</a>
</p>
