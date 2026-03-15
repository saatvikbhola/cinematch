"""CineMatch Embeddings — generates vector embeddings for movies and queries."""

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch
from config import EMBEDDING_MODEL

# Load model once (cached across calls)
_model = None

def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Model loaded ({_model.get_sentence_embedding_dimension()}d)")
    return _model


LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "ko": "Korean", "ja": "Japanese",
    "fr": "French", "es": "Spanish", "zh": "Chinese", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "th": "Thai", "tr": "Turkish", "pl": "Polish", "nl": "Dutch",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi",
    "ur": "Urdu", "id": "Indonesian", "ms": "Malay", "vi": "Vietnamese",
    "tl": "Filipino", "uk": "Ukrainian", "cs": "Czech", "ro": "Romanian",
    "hu": "Hungarian", "el": "Greek", "he": "Hebrew", "fa": "Persian",
    "cn": "Cantonese",
}

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

    # Year for temporal queries
    if movie.get("year") and movie["year"] > 0:
        parts.append(f"Year: {movie['year']}")

    return ". ".join(parts)


def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


# SPLADE sparse model
_splade_model = None
_splade_tokenizer = None

def _get_splade() -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """Lazy-load the SPLADE model."""
    global _splade_model, _splade_tokenizer
    if _splade_model is None:
        print("Loading SPLADE sparse model: naver/splade-cocondenser-ensembledistil")
        model_id = "naver/splade-cocondenser-ensembledistil"
        _splade_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _splade_model = AutoModelForMaskedLM.from_pretrained(model_id)
        if torch.backends.mps.is_available():
             _splade_model = _splade_model.to("mps")
        elif torch.cuda.is_available():
             _splade_model = _splade_model.to("cuda")
             
        _splade_model.eval()
        print("SPLADE loaded")
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

def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Generate embeddings for a batch of texts efficiently."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    result = []
    for e in embeddings:
        result.append(e.tolist())
    return result


def embed_movie(movie: dict) -> list[float]:
    """Generate embedding for a movie dict."""
    text = build_movie_text(movie)
    return embed_text(text)
