"""
CineMatch — AI Movie Discovery Engine 🎬
Powered by Endee Vector Database + Gemini 2.5 Flash

Run: streamlit run app.py
"""

import streamlit as st
import os
import sys
import time

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from search import search_with_filters, find_similar_by_text, get_db_stats
from ai_chat import explain_recommendations, get_chat_session, chat_followup, analyze_query_intent
from taste_profile import process_letterboxd_export
from rag import rag_answer

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="CineMatch — AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
)

# ============================================================================
# SESSION STATE INIT
# ============================================================================
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "ai_response" not in st.session_state:
    st.session_state.ai_response = ""
if "taste_profile" not in st.session_state:
    st.session_state.taste_profile = None
if "taste_search_query" not in st.session_state:
    st.session_state.taste_search_query = ""
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "search_time" not in st.session_state:
    st.session_state.search_time = 0.0
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []


# ============================================================================
# SIDEBAR — FILTERS + LETTERBOXD
# ============================================================================
with st.sidebar:
    st.header("🔑 API Keys")
    gemini_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_key,
        type="password",
        placeholder="AIzaSy...",
        help="Required for AI explanations and taste analysis. Get yours at https://aistudio.google.com/app/apikey"
    )
    if gemini_input != st.session_state.gemini_key:
        st.session_state.gemini_key = gemini_input
        st.rerun()
        
    if not st.session_state.gemini_key:
        st.warning("Please provide a Gemini API Key to unlock AI features.")
        
    st.divider()
    
    st.header("🎛️ Filters")

    genre_options = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History",
        "Horror", "Music", "Mystery", "Romance", "Science Fiction",
        "Thriller", "War", "Western"
    ]
    selected_genres = st.multiselect("Genres", genre_options, default=[])

    col1, col2 = st.columns(2)
    with col1:
        min_year = st.number_input("From Year", min_value=1900, max_value=2026, value=1900)
    with col2:
        max_year = st.number_input("To Year", min_value=1900, max_value=2026, value=2026)

    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, step=0.5)

    language_options = {"Any": None, "English": "en", "Korean": "ko", "Japanese": "ja",
                        "French": "fr", "Spanish": "es", "Hindi": "hi", "Chinese": "zh"}
    selected_lang = st.selectbox("Language", list(language_options.keys()))

    production_house = st.text_input("Production House", placeholder="e.g. Pixar, A24, Marvel")
    
    status_options = ["Any", "Released", "In Production", "Planned", "Post Production"]
    selected_status = st.selectbox("Movie Status", status_options)

    st.divider()
    
    st.header("🟢 Letterboxd Export")
    st.write("Upload your Letterboxd data for personalized recommendations.")

    ratings_file = st.file_uploader("ratings.csv", type="csv")
    reviews_file = st.file_uploader("reviews.csv", type="csv")


    if st.button("🧠 Analyze My Taste", width="stretch"):
        if ratings_file:
            with st.spinner("Analyzing your taste with Gemini..."):
                if not st.session_state.gemini_key:
                    st.error("Gemini API Key is required for Taste Analysis.")
                else:
                    result = process_letterboxd_export(
                        api_key=st.session_state.gemini_key,
                        ratings_content=ratings_file.getvalue().decode("utf-8") if ratings_file else None,
                        reviews_content=reviews_file.getvalue().decode("utf-8") if reviews_file else None
                    )
                    st.session_state.taste_profile = result["profile"]
                    st.session_state.taste_search_query = result["search_query"]
            st.success("✅ Taste profile generated!")
        else:
            st.warning("Please upload at least ratings.csv")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🎬 CineMatch")
st.markdown("##### *Discover movies by vibe, not just genre — powered by Endee Vector DB + Gemini*")
st.divider()

# Create tabs for the app
tab_search, tab_rag, tab_db = st.tabs(["🔍 Search & Discover", "🧠 RAG Q&A", "🗄️ Database Explorer"])

# ----------------------------------------------------------------------------
# TAB 1: SEARCH & DISCOVER
# ----------------------------------------------------------------------------
with tab_search:
    # --- Taste Profile Display ---
    if st.session_state.taste_profile:
        with st.expander("🧬 Your Taste Profile (from Letterboxd)", expanded=False):
            st.write(st.session_state.taste_profile)
            if st.session_state.taste_search_query:
                st.info(f"**Auto-generated search query:** {st.session_state.taste_search_query}")
                if st.button("🔍 Search with my taste profile"):
                    st.session_state.current_query = st.session_state.taste_search_query
                    st.rerun()

    # --- Search Bar ---
    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            "Search Box",
            placeholder="Describe a movie vibe... (e.g., 'a mind-bending thriller')",
            value=st.session_state.current_query,
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Search", width="stretch")

    # --- Handle Search ---
    if search_clicked and query:
        st.session_state.current_query = query
        
        # 1. AI Intent Analysis (if API key is present)
        ai_filters = {}
        search_query_text = query
        if st.session_state.gemini_key:
            with st.spinner("Extracting filters from query (AI)..."):
                ai_filters = analyze_query_intent(st.session_state.gemini_key, query)
                # Always use the original query for Endee — preserves the full vibe
                search_query_text = query
                
                # Show the user what we extracted
                extracted_details = [k for k in ai_filters.keys() if k != "refined_query"]
                if extracted_details:
                    st.caption(f"✨ AI extracted filters: {', '.join(extracted_details)}")

        # 2. Merge UI filters with AI-extracted filters
        # Genres
        merged_genres = set(selected_genres) if selected_genres else set()
        if "genres" in ai_filters and isinstance(ai_filters["genres"], list):
            merged_genres.update(ai_filters["genres"])
            
        # Production Companies
        merged_companies = set([production_house.strip()]) if production_house.strip() else set()
        if "production_companies" in ai_filters and isinstance(ai_filters["production_companies"], list):
            merged_companies.update(ai_filters["production_companies"])

        # Years and Ratings (UI overrides AI if explicitly set)
        final_min_year = min_year if min_year > 1900 else ai_filters.get("min_year")
        final_max_year = max_year if max_year < 2026 else ai_filters.get("max_year")
        final_min_rating = min_rating if min_rating > 0 else ai_filters.get("min_rating")
        
        # Language
        final_lang = language_options.get(selected_lang)
        if final_lang is None and "language" in ai_filters:
            final_lang = ai_filters["language"]

        with st.spinner("Searching Endee vector database..."):
            try:
                t0 = time.perf_counter()
                results = search_with_filters(
                    query=search_query_text,
                    genres=list(merged_genres) if merged_genres else None,
                    min_year=final_min_year,
                    max_year=final_max_year,
                    min_rating=final_min_rating,
                    language=final_lang,
                    production_companies=list(merged_companies) if merged_companies else None,
                    status=selected_status if selected_status != "Any" else None,
                    people=ai_filters.get("people"),
                    top_k=20,
                )
                st.session_state.search_time = time.perf_counter() - t0
                st.session_state.search_results = results
            except Exception as e:
                st.error(f"Search error: {e}")
                st.session_state.search_results = []
                st.session_state.search_time = 0.0

        if st.session_state.search_results:
            if st.session_state.gemini_key:
                with st.spinner("Gemini is explaining the picks..."):
                    try:
                        ai_resp = explain_recommendations(
                            api_key=st.session_state.gemini_key,
                            query=query,
                            movies=st.session_state.search_results,
                            taste_profile=st.session_state.taste_profile,
                        )
                        st.session_state.ai_response = ai_resp
                    except Exception as e:
                        st.session_state.ai_response = f"AI analysis unavailable: {e}"
            else:
                st.session_state.ai_response = "*(AI explanations disabled — add your Gemini API Key in the sidebar)*"

    # --- AI Response ---
    if st.session_state.ai_response:
        st.subheader("AI Picks for You")
        st.info(st.session_state.ai_response)

    # --- Search Results Grid ---
    if st.session_state.search_results:
        st.subheader(f"Results ({len(st.session_state.search_results)} movies) — {st.session_state.search_time:.2f}s")

        results = st.session_state.search_results
        cols_per_row = 4
        
        for row_start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, col in enumerate(cols):
                movie_idx = row_start + col_idx
                if movie_idx < len(results):
                    movie = results[movie_idx]
                    with col:
                        # Native Streamlit container for movie card
                        with st.container(border=True):
                            if movie.get("poster_url"):
                                st.image(movie["poster_url"], width="stretch")
                            else:
                                st.write("🎬 No Poster")
                                
                            st.markdown(f"**{movie.get('title', 'Unknown')}** ({movie.get('year', '')})")
                            st.caption(f"⭐ {movie.get('rating', 0)} | 🎯 Match: {movie.get('similarity', 0):.0%}")
                            st.caption(f"🎭 {movie.get('genres', '')}")
                            
                            with st.expander("Details"):
                                st.write(movie.get("overview", "No description available."))
                                st.write(f"**Director:** {movie.get('director', 'N/A')}")
                                st.write(f"**Cast:** {movie.get('cast', 'N/A')}")
                                st.write(f"**website:** https://www.themoviedb.org/movie/{movie.get('tmdb_id','N/A')}")
                                
                                if st.button("🔗 Find Similar", key=f"sim_{movie_idx}"):
                                    movie_text = f"{movie['title']}. {movie.get('overview', '')}. {movie.get('genres', '')}"
                                    with st.spinner("Finding similar movies..."):
                                        similar = find_similar_by_text(
                                            movie_text=movie_text,
                                            exclude_title=movie["title"],
                                            top_k=8,
                                        )
                                        if similar:
                                            st.session_state.search_results = similar
                                            st.session_state.ai_response = f"Showing movies similar to **{movie['title']}**"
                                            st.rerun()

    # --- Follow-up Chat ---
    if st.session_state.search_results:
        st.divider()
        st.subheader("💬 Refine with AI")
        
        followup = st.chat_input("Ask me to refine the results (e.g., 'something darker?')")
        if followup:
            with st.chat_message("user"):
                st.write(followup)
                
            with st.chat_message("assistant"):
                if not st.session_state.gemini_key:
                    st.warning("Please add your Gemini API Key in the sidebar to chat.")
                else:
                    with st.spinner("Thinking..."):
                        response = chat_followup(
                            api_key=st.session_state.gemini_key,
                            message=followup,
                            movies_context=st.session_state.search_results,
                            taste_profile=st.session_state.taste_profile,
                        )
                        st.write(response)

    # --- Empty State ---
    if not st.session_state.search_results and not st.session_state.ai_response:
        st.divider()
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "<h4>Describe your perfect movie!</h4>"
            "<p>Try: 'a mind-bending thriller' or 'feel-good comedy'</p>"
            "</div>", 
            unsafe_allow_html=True
        )

# ----------------------------------------------------------------------------
# TAB 2: RAG Q&A
# ----------------------------------------------------------------------------
with tab_rag:
    st.subheader("🧠 Ask Anything About Movies")
    st.markdown(
        "Ask a natural-language question and get a **grounded answer** powered by "
        "Endee vector retrieval + Gemini generation. Every answer cites real movies from the database."
    )
    st.divider()

    # Display conversation history
    for entry in st.session_state.rag_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            with st.expander(f"📚 Retrieved Context ({entry['num_retrieved']} movies)"):
                for i, m in enumerate(entry["retrieved_movies"], 1):
                    st.markdown(
                        f"**[{i}] {m['title']}** ({m.get('year', '')}) — "
                        f"⭐ {m.get('rating', 'N/A')} | 🎯 {m.get('similarity', 0):.0%} match\n\n"
                        f"🎭 {m.get('genres', '')} | 🎬 {m.get('director', 'N/A')}"
                    )

    # Chat input
    rag_question = st.chat_input(
        "Ask a movie question... (e.g., 'I want to try getting into korean cinema, where should i start from ?')",
        key="rag_input",
    )

    if rag_question:
        with st.chat_message("user"):
            st.write(rag_question)

        with st.chat_message("assistant"):
            if not st.session_state.gemini_key:
                st.warning("Please add your Gemini API Key in the sidebar to use RAG Q&A.")
            else:
                with st.spinner("🔎 Retrieving from Endee & generating answer..."):
                    result = rag_answer(
                        api_key=st.session_state.gemini_key,
                        question=rag_question,
                        genres=selected_genres if selected_genres else None,
                        min_year=min_year if min_year > 1900 else None,
                        max_year=max_year if max_year < 2026 else None,
                        min_rating=min_rating if min_rating > 0 else None,
                        language=language_options.get(selected_lang),
                        production_companies=[production_house.strip()] if production_house.strip() else None,
                        status=selected_status if selected_status != "Any" else None,
                        taste_profile=st.session_state.taste_profile,
                        taste_search_query=st.session_state.taste_search_query,
                        top_k=8,
                    )
                st.markdown(result["answer"])

                if result["retrieved_movies"]:
                    with st.expander(f"📚 Retrieved Context ({result['num_retrieved']} movies)"):
                        for i, m in enumerate(result["retrieved_movies"], 1):
                            st.markdown(
                                f"**[{i}] {m['title']}** ({m.get('year', '')}) — "
                                f"⭐ {m.get('rating', 'N/A')} | 🎯 {m.get('similarity', 0):.0%} match\n\n"
                                f"🎭 {m.get('genres', '')} | 🎬 {m.get('director', 'N/A')}"
                            )

                # Save to history
                st.session_state.rag_history.append({
                    "question": rag_question,
                    "answer": result["answer"],
                    "retrieved_movies": result["retrieved_movies"],
                    "num_retrieved": result["num_retrieved"],
                })

    # Empty state
    if not st.session_state.rag_history:
        st.info(
            "**Ask me anything about movies! 🎬**\n\n"
            "Examples:\n"
            "- *'What's a good movie for a first date?'*\n"
            "- *'Recommend a dark psychological thriller'*\n"
            "- *'What are some feel-good animated films?'*"
        )

# ----------------------------------------------------------------------------
# TAB 3: DATABASE EXPLORER
# ----------------------------------------------------------------------------
with tab_db:
    st.subheader("🗄️ Endee Database Metrics")
    
    with st.spinner("Fetching database statistics..."):
        stats = get_db_stats()
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Indexed Movies", f"{stats.get('total_movies', 0):,}")
    col2.metric("Vector Dimension", "384")
    col3.metric("Similarity Metric", "Cosine")
    
    st.divider()
    st.subheader("📂 Browse Movies")
    st.write("Search the database to browse indexed movies.")
    
    browse_query = st.text_input("Browse by keyword", placeholder="e.g., action, comedy, sci-fi...", key="browse_input")
    if st.button("🔄 Browse", key="refresh_db") and browse_query:
        with st.spinner("Searching..."):
            sample_movies = search_with_filters(
                query=browse_query,
                genres=selected_genres if selected_genres else None,
                min_year=min_year if min_year > 1900 else None,
                max_year=max_year if max_year < 2026 else None,
                min_rating=min_rating if min_rating > 0 else None,
                language=language_options.get(selected_lang),
                production_companies=[production_house.strip()] if production_house.strip() else None,
                status=selected_status if selected_status != "Any" else None,
            )
        
        if sample_movies:
            import pandas as pd
            
            table_data = []
            for m in sample_movies:
                table_data.append({
                    "Title": m.get("title", ""),
                    "Year": m.get("year", ""),
                    "Rating": f"⭐ {m.get('rating', 0)}",
                    "Genres": m.get("genres", ""),
                    "Director": m.get("director", ""),
                    "Language": m.get("language", "").upper(),
                    "DB ID": m.get("id", "")
                })
                
            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    "Title": st.column_config.TextColumn("Movie Title", width="large"),
                    "Year": st.column_config.NumberColumn("Year", format="%d"),
                },
                width="stretch",
                height=600
            )
        else:
            st.info("No results found. Try a different keyword.")
