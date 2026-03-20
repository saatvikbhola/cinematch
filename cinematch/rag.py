"""
CineMatch RAG Pipeline — True Retrieval-Augmented Generation.

Accepts a natural-language question, retrieves relevant movies from Endee
via hybrid vector search, and generates a grounded answer with citations
using Gemini.
"""

import time

from config import ENDEE_URL, ENDEE_INDEX_NAME
from embeddings import embed_text, get_sparse_embedding
from search import _format_results
from endee import Endee
from ai_chat import call_llm

RAG_SYSTEM_PROMPT = """You are CineMatch RAG, a movie expert that ONLY answers 
based on retrieved movie data. You must NEVER invent or hallucinate information.

Rules:
- Answer the user's question using ONLY the retrieved movies provided below.
- Cite specific movies by their [number] reference, e.g. [1], [2].
- DETECTION: If the user mentions a specific movie (e.g., "I Saw the TV Glow") and it is NOT in the retrieved list, EXPLICITLY state that you don't have that specific movie in your database.
- HONESTY: If the retrieved movies are not a good fit for the user's request (e.g., similarity scores are low or the content is unrelated), state clearly that you couldn't find a strong match.
- PERSONALIZATION: If a user taste profile is provided, use it to personalize your tone and explain why specifically THESE movies fit THEIR taste beyond just the query.
- Be conversational, concise, and engaging.
- Format your response with bullet points for readability.
- Include relevant details like year, rating, director, and genres when citing.
"""


def _get_index():
    """Get the Endee movies index."""
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")
    return client.get_index(name=ENDEE_INDEX_NAME)


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
    """
    Retrieve relevant movies from Endee using hybrid search, optionally personalized.

    Args:
        question: Natural language question from the user.
        taste_search_query: Optional search query from user's taste profile.
        top_k: Number of movies to retrieve.

    Returns:
        List of formatted movie dicts from Endee.
    """
    # Personalization fallback: if no question, use taste query. 
    # Otherwise, prioritize question but incorporate taste keywords subtly.
    search_text = question
    if taste_search_query:
        # We combine them so hybrid search considers both the vibe search and the specific question
        search_text = f"{question}. {taste_search_query}"

    # Generate dense embedding for semantic similarity
    query_vector = embed_text(search_text)

    # Generate sparse embedding for keyword matching
    sparse_indices, sparse_values = get_sparse_embedding(search_text)

    # --- DEBUG: Raw Log ---
    print(f"\nRAG RETRIEVAL")
    print(f"   Question: '{question}'")
    if taste_search_query:
        print(f"   Taste Query: '{taste_search_query[:50]}...'")
    print(f"   Dense dim: {len(query_vector)} | Sparse indices: {len(sparse_indices)}")

    index = _get_index()

    # Build filter list for Endee using server-side $eq and $range.
    # Genre and company filters use dynamic boolean flags (e.g., genre_action: "yes").
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

    # Default rating filter if none provided, otherwise use provided min_rating
    actual_min_rating = min_rating if min_rating is not None else 5.0
    filters.append({"rating": {"$range": [actual_min_rating, 10.0]}})
    
    # Server-side matching for dynamic genre flags
    if genres:
        for g in genres:
            safe_genre = g.lower().replace(" ", "_")
            filters.append({f"genre_{safe_genre}": {"$eq": "yes"}})

    # Server-side matching for dynamic company flags
    if production_companies:
        for pc in production_companies:
            safe_company = pc.lower().replace(" ", "_")
            filters.append({f"company_{safe_company}": {"$eq": "yes"}})

    # Always over-fetch to ensure a deep semantic pool for client-side filtering
    # Studio names like "A24" are weak semantic signals compared to movie plots
    fetch_k = max(100, top_k * 5)

    print(f"   Endee Filter: {filters if filters else None}")

    try:
        t0 = time.perf_counter()
        results = index.query(
            vector=query_vector, 
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=fetch_k, 
            filter=filters if filters else None,
        )
        print(f"   Fetched {len(results)} results from Endee (Hybrid)")
    except Exception as e:
        print(f"Hybrid query failed ({e}), trying dense-only...")
        try:
            t0 = time.perf_counter()
            results = index.query(
                vector=query_vector,
                top_k=fetch_k,
                filter=filters if filters else None,
            )
            print(f"   Fetched {len(results)} results from Endee (Dense Fallback)")
        except Exception as e2:
            print(f"Dense query also failed ({e2})")
            results = []

    formatted = _format_results(results)
    formatted = formatted[:top_k]
    print(f"   Retrieved {len(formatted)} movies from Endee")
    return formatted


def _build_context(movies: list[dict]) -> str:
    """Build a numbered context block from retrieved movies for the LLM."""
    context_parts = []
    for i, m in enumerate(movies, 1):
        context_parts.append(
            f"[{i}] {m['title']} ({m.get('year', 'N/A')})\n"
            f"    Rating: {m.get('rating', 'N/A')}/10\n"
            f"    Genres: {m.get('genres', 'N/A')}\n"
            f"    Director: {m.get('director', 'N/A')}\n"
            f"    Cast: {m.get('cast', 'N/A')}\n"
            f"    Studio: {m.get('production_companies', 'N/A')}\n"
            f"    Plot: {m.get('overview', 'No overview available.')}\n"
            f"    Similarity: {m.get('similarity', 0):.0%}"
        )
    return "\n\n".join(context_parts)


def reformulate_rag_query(
    api_key: str,
    question: str,
    history: list[dict] | None = None,
    provider: str = "Gemini"
) -> str:
    """
    Reformulate a follow-up question into a standalone, descriptive search query.
    If no history is provided or it's not a follow-up, returns the original question.
    """
    if not history:
        return question

    # Build history context for the LLM
    context = ""
    for entry in history[-3:]: # Limit to last 3 turns for context
        context += f"User: {entry['question']}\nAI: {entry['answer'][:200]}...\n"

    prompt = f"""Given the following conversation history and a new follow-up question, 
reformulate the question into a SINGLE standalone search query that can be used to find movies 
in a vector database. 

The standalone query should:
1. Be descriptive and capture the full intent (including context from history).
2. Avoid pronouns like "them", "those", "that" — replace them with specific movie names or themes from the history.
3. Focus on the vibe, genre, or specific details the user is now looking for.

CONVERSATION HISTORY:
{context}

FOLLOW-UP QUESTION:
"{question}"

STANDALONE QUERY:"""

    try:
        response_data = call_llm(
            api_key, 
            prompt, 
            system_instruction="You are a search query optimizer. Return ONLY the reformulated query text.", 
            provider=provider
        )
        return response_data["text"].strip().strip('"')
    except Exception:
        return question


def rag_answer(
    api_key: str,
    question: str,
    provider: str = "Gemini",
    history: list[dict] | None = None,
    genres: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    language: str | None = None,
    production_companies: list[str] | None = None,
    status: str | None = None,
    taste_profile: str | None = None,
    taste_search_query: str = "",
    top_k: int = 8,
) -> dict:
    """
    Full RAG pipeline with query reformulation: 
    Reformulate → Retrieve from Endee → Generate grounded answer with LLM.

    Args:
        api_key: User's AI API key.
        question: Natural language question (could be a follow-up).
        provider: AI provider (Gemini or OpenRouter Free).
        history: Previous RAG conversation history.
        taste_profile: Optional taste profile description text.
        taste_search_query: Optional taste-based semantic query.
        top_k: Number of movies to retrieve for context.
    """
    if not api_key:
        return {
            "answer": f"Please provide your {provider} API key in the sidebar to use RAG Q&A.",
            "retrieved_movies": [],
            "num_retrieved": 0,
        }

    # Step 1: Reformulate query if there is history
    standalone_query = reformulate_rag_query(api_key, question, history, provider)
    print(f"   Reformulated Query: '{standalone_query}'")

    # Step 2: Retrieve relevant movies from Endee using reformulated query
    retrieved = retrieve_from_endee(
        standalone_query, 
        genres=genres,
        min_year=min_year,
        max_year=max_year,
        min_rating=min_rating,
        language=language,
        production_companies=production_companies,
        status=status,
        taste_search_query=taste_search_query, 
        top_k=top_k
    )

    if not retrieved:
        return {
            "answer": "I couldn't find any relevant movies in the database for your question. Try rephrasing or broadening your query.",
            "retrieved_movies": [],
            "num_retrieved": 0,
        }

    # Step 3: Build structured context from retrieved movies
    context = _build_context(retrieved)

    # Step 4: Generate grounded answer
    history_context = ""
    if history:
        history_context = "\nCONVERSATION HISTORY:\n"
        for entry in history[-3:]:
            history_context += f"User: {entry['question']}\nAI: {entry['answer'][:300]}...\n"

    taste_context = ""
    if taste_profile:
        taste_context = f"\nUser's Taste Profile (from Letterboxd):\n{taste_profile}\n---\n"

    prompt = f"""{taste_context}{history_context}
CURRENT QUESTION: "{question}"

Retrieved Movies from Endee Vector Database (ranked by semantic similarity):

{context}

---

Using ONLY the retrieved movies above, answer the current question. 
If the question is a follow-up, use the history for context but prioritize answering the new question.
Cite movies by their [number] reference (e.g., [1], [2]).
Explain WHY each recommended movie fits the question."""

    try:
        response_data = call_llm(api_key, prompt, system_instruction=RAG_SYSTEM_PROMPT, provider=provider)
        answer = response_data["text"]
        model_used = response_data["model"]
    except Exception as e:
        answer = f"Error generating answer: {e}"
        model_used = "error"

    return {
        "answer": answer,
        "retrieved_movies": retrieved,
        "num_retrieved": len(retrieved),
        "standalone_query": standalone_query,
        "model": model_used
    }
