import google.generativeai as genai
import litellm
import json
import os

from config import GEMINI_MODEL, OPENROUTER_BASE_URL, OPENROUTER_DEFAULT_MODEL

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


def call_llm(
    api_key: str,
    prompt: str,
    system_instruction: str = SYSTEM_PROMPT,
    provider: str = "Gemini",
    history: list[dict] | None = None,
) -> dict:
    """Unified LLM caller for Gemini and OpenRouter using LiteLLM."""
    if not api_key:
        raise ValueError(f"{provider} API key is required")

    if provider == "Gemini":
        genai.configure(api_key=api_key)
        model_name = GEMINI_MODEL
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        
        gemini_history = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})
        
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(prompt)
        return {"text": response.text, "model": model_name}

    elif provider == "OpenRouter Free":
        # Set OpenRouter API key for LiteLLM
        os.environ["OPENROUTER_API_KEY"] = api_key
        
        messages = [{"role": "system", "content": system_instruction}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = litellm.completion(
                model=f"openrouter/{OPENROUTER_DEFAULT_MODEL}",
                messages=messages,
                base_url=OPENROUTER_BASE_URL,
            )
            
            # LiteLLM returns the specific model in response.model
            model_used = response.get("model", OPENROUTER_DEFAULT_MODEL)
            content = response.choices[0].message.content
            
            return {"text": content, "model": model_used}
        except Exception as e:
            raise Exception(f"LiteLLM/OpenRouter Error: {e}")
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def explain_recommendations(
    api_key: str,
    query: str,
    movies: list[dict],
    provider: str = "Gemini",
    taste_profile: str | None = None,
    chat_history: list | None = None,
) -> str:
    """Generate AI explanation for why these movies match the user's query."""
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
        response_data = call_llm(api_key, prompt, provider=provider, history=chat_history)
        return response_data # Returns dict with 'text' and 'model'
    except Exception as e:
        return {"text": f"Hmm, I couldn't generate recommendations right now. Error: {e}", "model": "error"}


def chat_followup(
    api_key: str,
    message: str,
    provider: str = "Gemini",
    movies_context: list[dict] | None = None,
    taste_profile: str | None = None,
    chat_history: list | None = None,
) -> str:
    """Handle follow-up chat messages for refining recommendations."""
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
        response_data = call_llm(api_key, prompt, provider=provider, history=chat_history)
        return response_data
    except Exception as e:
        return {"text": f"Sorry, I hit an issue: {e}", "model": "error"}


def analyze_query_intent(api_key: str, query: str, provider: str = "Gemini") -> dict:
    """Use AI to understand user's search intent and extract filters."""
    if not api_key:
        return {"refined_query": query}
        
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
        response_data = call_llm(api_key, prompt, provider=provider, system_instruction="You are a JSON generator. Return ONLY valid JSON.")
        text = response_data["text"].strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:].strip()

        # We return the model along with refined info
        result = json.loads(text)
        result["_model"] = response_data["model"]
        return result
    except Exception:
        return {"refined_query": query}
