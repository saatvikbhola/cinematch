import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is not found in the .env file.")
    exit(1)

print(f"Found Gemini API Key starting with: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("Sending test prompt to Gemini (gemini-2.5-flash)...")
    response = model.generate_content("Say 'Hello, API works!' if you receive this.")
    print("\nSUCCESS! Gemini responded:")
    print("-" * 30)
    print(response.text.strip())
    print("-" * 30)
except Exception as e:
    print(f"\nERROR connecting to Gemini API:\n {e}")
