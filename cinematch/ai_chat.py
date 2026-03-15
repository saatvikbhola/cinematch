"""CineMatch AI Chat — Gemini 2.5 Pro powered conversational movie assistant."""

import google.generativeai as genai

from config import GEMINI_MODEL

# System prompt for the movie assistant
SYSTEM_PROMPT = """You are CineMatch AI, a passionate and knowledgeable film expert. You help users discover movies from our curated database.

Your personality:
- You are a thoughtful, analytical curator of film.
- You tailor your tone to the user's request: serious for dramas, light for comedies.
- You give concise, well-reasoned recommendations without overusing exclamation points.
- You reference specific aspects: cinematography, soundtracks, performances, themes.
- You use emojis sparingly and only for structural emphasis (e.g., bullet points).
- You format responses with bullet points for readability.

When recommending movies:
- Always explain WHY a movie matches what the user is looking for
- Reference specific scenes, themes, or qualities (without major spoilers)
- If you know their taste profile, tailor recommendations to their preferences
- Suggest 3-5 movies per recommendation, not more

When analyzing taste profiles:
- Be perceptive and specific — don't just list genres
- Reference their actual ratings and reviews
- Identify patterns they might not see themselves

CRITICAL RULE: You MUST ONLY recommend and discuss movies from the search results provided to you. NEVER suggest, mention, or recommend movies that are not in the provided search results. If the results don't match the user's request well, be honest about it — say the database didn't have a great match — but do NOT invent or add your own movie suggestions. Our database is the single source of truth."""


def get_chat_session(api_key: str, history: list[dict] | None = None):
    """Create or resume a Gemini chat session."""
    if not api_key:
        raise ValueError("Gemini API key is required")
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    if history:
        return model.start_chat(history=history)
    return model.start_chat()


def explain_recommendations(
    api_key: str,
    query: str,
    movies: list[dict],
    taste_profile: str | None = None,
    chat_history: list | None = None,
) -> str:
    """
    Generate AI explanation for why these movies match the user's query.

    Args:
        api_key: User's Gemini API key
        query: The user's original search query
        movies: List of movie dicts from Endee search results
        taste_profile: Optional user taste profile from Letterboxd
        chat_history: Previous chat messages for context

    Returns:
        Gemini-generated explanation/recommendation text
    """
    try:
        chat = get_chat_session(api_key, chat_history)
    except ValueError as e:
        return f"Please set your Gemini API Key in the sidebar to get AI recommendations."

    # Build movie context
    movie_context = ""
    for i, m in enumerate(movies[:8], 1):
        movie_context += f"""
{i}. **{m['title']}** ({m['year']}) — ⭐ {m['rating']}/10
   Genres: {m['genres']}
   Director: {m['director']}
   Studio: {m.get('production_companies', 'N/A')}
   Cast: {m['cast']}
   Plot: {m['overview'][:200]}...
   Match Score: {m['similarity']:.0%}
"""

    prompt = f"""The user searched for: "{query}"

Here are the movies found from our vector database (ranked by semantic similarity):
{movie_context}
"""

    if taste_profile:
        prompt += f"""
The user's taste profile (from their Letterboxd history):
{taste_profile[:500]}
"""

    prompt += """
Based on these search results and the user's query, give a conversational recommendation. Highlight the top 3-5 picks and explain why each matches what they're looking for. Be concise and engaging."""

    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Hmm, I couldn't generate recommendations right now. Error: {e}"


def chat_followup(
    api_key: str,
    message: str,
    movies_context: list[dict] | None = None,
    taste_profile: str | None = None,
    chat_history: list | None = None,
) -> str:
    """
    Handle follow-up chat messages for refining recommendations.

    Args:
        api_key: User's Gemini API key
        message: User's follow-up message
        movies_context: Current movie results for reference
        taste_profile: Optional taste profile
        chat_history: Previous chat messages

    Returns:
        Gemini response
    """
    try:
        chat = get_chat_session(api_key, chat_history)
    except ValueError as e:
        return "Please set your Gemini API Key in the sidebar first."

    prompt = message

    if movies_context:
        context = "\n".join([
            f"- {m['title']} ({m['year']}) — {m['genres']} | Studio: {m.get('production_companies', 'N/A')}"
            for m in movies_context[:10]
        ])
        prompt = f"""Context — movies currently shown to the user from our database:
{context}

User's message: {message}

Respond naturally. You may ONLY discuss the movies listed above — do NOT suggest any movies outside this list. If they want different results, suggest refining their search query or adjusting filters. If they ask about a specific movie from the list, give details."""

    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I hit an issue: {e}"


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
  3. Keep ALL movie/director/actor references (e.g., 'like Back to the Future', 'Wes Anderson style', 'Tarantino vibe')
  4. Keep ALL thematic descriptors (e.g., 'time travel', 'heist', 'coming of age', 'revenge')
  5. ONLY remove pure filter tokens: explicit year numbers (e.g., '2010s', 'from 2020'), rating numbers (e.g., 'rated above 7'), and language names when used as a filter (e.g., 'in Korean')
  6. If in doubt, KEEP the word. Over-preserving is always better than over-stripping.

  Example: "cyberpunk future type of Back to the Future" → refined_query: "cyberpunk future type of Back to the Future" (keep everything, nothing to extract)
  Example: "highly rated Korean thrillers from the 2010s" → refined_query: "intense thrillers" (removed 'Korean' → language filter, '2010s' → year filter, 'highly rated' → rating filter)

- "genres": list of genres if mentioned (use TMDb genre names: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western)
- "production_companies": list of studios or production companies if mentioned (e.g., A24, Pixar, Marvel Studios)
- "people": list of specific people (actors, directors, writers) mentioned in the query (e.g., ["Timothée Chalamet", "Christopher Nolan", "Zendaya"]).
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
