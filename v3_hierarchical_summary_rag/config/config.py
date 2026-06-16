# config.py
import os
import psycopg2
from dotenv import load_dotenv
import google.genai as genai

# Load environment configuration variables by looking up two levels to the root directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

# Initialize the official central Google GenAI Client
client = genai.Client(http_options={'timeout': 60.0})

def get_db_connection():
    """Returns a raw active connection instance to the PostgreSQL container."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
