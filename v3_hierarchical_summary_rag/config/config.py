import os
import sys
import psycopg2
import google.genai as genai
from dotenv import load_dotenv

# 1. Load System Environment
load_dotenv(override=True)

# 2. Configure the Google Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ CRITICAL ERROR: GEMINI_API_KEY is missing from your .env file.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# 3. Configure Hugging Face API Key
HF_KEY = os.getenv("HF_KEY")
if not HF_KEY:
    print("⚠️ WARNING: HF_KEY is missing. Reranking will be disabled/fallback.")

# 4. Database Connection Pooler
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
    except Exception as e:
        print(f"❌ DATABASE CONNECTION FAILED: {e}")
        sys.exit(1)