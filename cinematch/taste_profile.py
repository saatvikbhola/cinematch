"""
CineMatch Taste Profile — Parses Letterboxd export data to build a user taste profile.
Uses Gemini to analyze viewing patterns and generate a "taste DNA" description.
"""

import csv
import io
import os
import re

import google.generativeai as genai

from config import GEMINI_MODEL


def parse_ratings(file_content: str) -> list[dict]:
    """Parse Letterboxd ratings.csv content."""
    reader = csv.DictReader(io.StringIO(file_content))
    movies = []
    for row in reader:
        if row.get("Name"):
            movies.append({
                "name": row["Name"].strip(),
                "year": row.get("Year", "").strip(),
                "rating": float(row.get("Rating", 0) or 0),
            })
    return movies


def parse_reviews(file_content: str) -> list[dict]:
    """Parse Letterboxd reviews.csv content."""
    reader = csv.DictReader(io.StringIO(file_content))
    reviews = []
    for row in reader:
        if row.get("Name") and row.get("Review"):
            review_text = row["Review"].strip()
            # Clean HTML tags
            review_text = re.sub(r"<[^>]+>", "", review_text)
            reviews.append({
                "name": row["Name"].strip(),
                "year": row.get("Year", "").strip(),
                "rating": float(row.get("Rating", 0) or 0),
                "review": review_text,
            })
    return reviews


def parse_diary(file_content: str) -> list[dict]:
    """Parse Letterboxd diary.csv content."""
    reader = csv.DictReader(io.StringIO(file_content))
    entries = []
    for row in reader:
        if row.get("Name"):
            entries.append({
                "name": row["Name"].strip(),
                "year": row.get("Year", "").strip(),
                "rating": float(row.get("Rating", 0) or 0),
                "watched_date": row.get("Watched Date", "").strip(),
                "rewatch": row.get("Rewatch", "").strip().lower() == "yes",
            })
    return entries


def parse_watchlist(file_content: str) -> list[dict]:
    """Parse Letterboxd watchlist.csv content."""
    reader = csv.DictReader(io.StringIO(file_content))
    movies = []
    for row in reader:
        if row.get("Name"):
            movies.append({
                "name": row["Name"].strip(),
                "year": row.get("Year", "").strip(),
            })
    return movies


def build_taste_summary(
    ratings: list[dict],
    reviews: list[dict],
    diary: list[dict],
    watchlist: list[dict],
) -> dict:
    """Build a raw taste summary from parsed Letterboxd data."""
    # Categorize by rating
    loved = [m for m in ratings if m["rating"] >= 4]
    liked = [m for m in ratings if 3 <= m["rating"] < 4]
    disliked = [m for m in ratings if m["rating"] < 3 and m["rating"] > 0]

    return {
        "total_watched": len(ratings),
        "loved": loved,
        "liked": liked,
        "disliked": disliked,
        "reviews": reviews,
        "watchlist": watchlist,
        "recent": sorted(diary, key=lambda x: x.get("watched_date", ""), reverse=True)[:10],
    }


def generate_taste_profile(api_key: str, taste_summary: dict) -> str:
    """
    Use Gemini to analyze viewing history and generate a taste profile description.
    This description is then used as a search query in Endee to find matching movies.
    """
    if not api_key:
        return "Please provide a Gemini API Key to analyze your taste profile."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Build the prompt
    loved_str = "\n".join([f"  - {m['name']} ({m['year']}) — ⭐ {m['rating']}/5" for m in taste_summary["loved"]])
    disliked_str = "\n".join([f"  - {m['name']} ({m['year']}) — ⭐ {m['rating']}/5" for m in taste_summary["disliked"]])
    reviews_str = "\n".join([f"  - {r['name']}: \"{r['review']}\" (⭐ {r['rating']}/5)" for r in taste_summary["reviews"]])
    watchlist_str = "\n".join([f"  - {m['name']} ({m['year']})" for m in taste_summary["watchlist"]])

    prompt = f"""You are a film critic and taste analyst. Analyze this person's Letterboxd viewing history and create a detailed taste profile.

## Movies They LOVED (4-5 stars):
{loved_str or "No data"}

## Movies They DISLIKED (1-2 stars):
{disliked_str or "No data"}

## Their Reviews:
{reviews_str or "No reviews"}

## Their Watchlist (want to see):
{watchlist_str or "Empty"}

## Your Task:
Write a detailed 2-3 paragraph taste profile that captures:
1. What genres, themes, and styles they gravitate toward
2. What they value in films (visuals, storytelling, acting, etc.)
3. What they tend to dislike
4. Their overall cinematic personality

Be specific and perceptive. Reference their actual movies and reviews to support your analysis. Write it in second person ("You tend to...").

Then, on a new line after "SEARCH_QUERY:", write a single search query (2-3 sentences) that could be used to find movies they would love. This should capture the essence of their taste in descriptive, semantic terms."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate taste profile: {e}"


def extract_search_query(profile_text: str) -> str:
    """Extract the SEARCH_QUERY from the generated taste profile."""
    if "SEARCH_QUERY:" in profile_text:
        query = profile_text.split("SEARCH_QUERY:")[-1].strip()
        return query
    # Fallback: use the last paragraph
    paragraphs = [p.strip() for p in profile_text.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else ""


def process_letterboxd_export(
    api_key: str,
    ratings_content: str | None = None,
    reviews_content: str | None = None,
    diary_content: str | None = None,
    watchlist_content: str | None = None,
) -> dict:
    """
    Full pipeline: parse Letterboxd CSVs → summarize → Gemini taste analysis.

    Returns:
        {
            "summary": raw taste summary dict,
            "profile": Gemini-generated taste profile text,
            "search_query": extracted search query for Endee
        }
    """
    ratings = parse_ratings(ratings_content) if ratings_content else []
    reviews = parse_reviews(reviews_content) if reviews_content else []
    diary = parse_diary(diary_content) if diary_content else []
    watchlist = parse_watchlist(watchlist_content) if watchlist_content else []

    summary = build_taste_summary(ratings, reviews, diary, watchlist)
    profile = generate_taste_profile(api_key, summary)
    search_query = extract_search_query(profile)

    return {
        "summary": summary,
        "profile": profile,
        "search_query": search_query,
    }
