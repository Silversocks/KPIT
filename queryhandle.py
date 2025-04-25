import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CHUNK_FILE = "chunks.json"
MODEL_NAME = "models/gemini-2.0-flash"  # Change this if you're using a different version
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

def load_chunks():
    with open(CHUNK_FILE, "r") as f:
        return json.load(f)

def find_relevant_chunks(query, chunks):
    return [chunk["content"] for chunk in chunks if query.lower() in chunk["content"].lower()]

def query_model(error_code):
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set as an environment variable.")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    chunks = load_chunks()

    query_text = f"What does the error code {error_code} mean in a car?"
    matched_chunks = find_relevant_chunks(error_code, chunks)

    if not matched_chunks:
        return "Sorry, I couldn’t find anything about that error code."

    context = "\n\n".join(matched_chunks)
    prompt = f"{context}\n\nBased on the above data, generate a detailed report regarding the error codes {error_code} in a car. Include the problem, potential causes and fixes. Include citations for points made, and tailor the report for a non mechanic driver of the vehicle."

    response = model.generate_content(prompt)
    return response.text.strip()

def main():
    error_code = input("📟 Enter OBD Error Code (e.g., P0522): ").strip().upper()
    result = query_model(error_code)
    print("\n🚘 Car Screen Output:")
    print(result)

if __name__ == "__main__":
    main()