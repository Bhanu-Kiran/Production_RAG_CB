import os
import psycopg2
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables by looking up one level to the root directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Initialize official GenAI Client
client = genai.Client()

def get_db_connection():
    """Establishes connection to the local PostgreSQL database."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def get_google_embedding(text):
    """Generates a 768-dimensional semantic vector for the input text."""
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"   ❌ [Error] Embedding generation failed: {e}")
        return None

def retrieve_hybrid_context(search_query, query_vector, limit=10):
    """
    Executes a Hybrid Search (Semantic Vector + Full-Text Keyword Search)
    inside PostgreSQL and merges rankings using Reciprocal Rank Fusion (RRF).
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Comprehensive Hybrid SQL Query using Common Table Expressions (CTEs)
        hybrid_sql = """
        -- CTE 1: THE VECTOR SEARCH
        -- We rank the top 20 documents conceptually closest to the question using the <=> Cosine Distance operator.
        WITH vector_search AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as rank
            FROM document_elements
            ORDER BY embedding <=> %s::vector
            LIMIT 20
        ),
        
        -- CTE 2: THE KEYWORD SEARCH
        -- We use plainto_tsquery to safely convert the raw query into searchable keyword tokens.
        -- We rank the top 20 documents that contain exact word matches.
        keyword_search AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC) as rank
            FROM document_elements
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            LIMIT 20
        )
        
        -- CTE 3: THE MERGER (Reciprocal Rank Fusion)
        -- We join the two temporary tables together based on the Document ID.
        -- We apply the RRF math formula: 1 / (60 + Rank). 
        -- COALESCE handles cases where a document is found in one list but missing from the other.
        SELECT 
            d.file_name, 
            e.content,
            COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) as rrf_score
        FROM document_elements e
        JOIN production_documents d ON e.document_id = d.id
        LEFT JOIN vector_search v ON e.id = v.id
        LEFT JOIN keyword_search k ON e.id = k.id
        WHERE v.id IS NOT NULL OR k.id IS NOT NULL
        ORDER BY rrf_score DESC
        LIMIT %s;
        """
        
        # Execute query. We pass query_vector twice for CTE 1, search_query twice for CTE 2.
        cursor.execute(hybrid_sql, (query_vector, query_vector, search_query, search_query, limit))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return results
        
    except Exception as e:
        print(f"   ⚠️ [Hybrid Search Notice] Text index failed, falling back to basic vector search... ({e})")
        return fallback_pure_vector_search(query_vector, limit)

def fallback_pure_vector_search(query_vector, limit):
    """Fallback function to ensure the system stays online if text parsing errors occur."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.file_name, e.content, 1 - (e.embedding <=> %s::vector) as rrf_score
        FROM document_elements e
        JOIN production_documents d ON e.document_id = d.id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """, (query_vector, query_vector, limit))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

if __name__ == "__main__":
    test_query = "What is the ICU room limit in the SBI policy?"
    print(f"=============================================================")
    print(f"🔎 ADVANCED RETRIEVAL ENGINE DIAGNOSTIC")
    print(f"   Query: \"{test_query}\"")
    print(f"=============================================================")
    
    print("🧠 Embedding query...")
    vector = get_google_embedding(test_query)
    
    if vector:
        print("🔍 Searching database using Hybrid RRF math...")
        results = retrieve_hybrid_context(test_query, vector, limit=3)
        
        print("\n--- TOP MATCHES ---")
        for idx, row in enumerate(results, 1):
            file_name, content, score = row
            snippet = " ".join(content.replace("\n", " ").split())[:150]
            print(f"\n{idx}. Source File: {file_name} | RRF Score: {score:.5f}")
            print(f"   Snippet: \"{snippet}...\"")
    else:
        print("❌ Embedding extraction failed.")