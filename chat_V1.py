# A basic pipeline for a RAG system that uses pgvector for semantic search and Google Gemini for text generation.
# It uses the sentence-transformers library to generate embeddings for the input query.

import os
import psycopg2
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 1. INITIALIZATION & SETUP
# ---------------------------------------------------------------------
load_dotenv()
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

# def retrieve_context_only_Semantic(query_vector, limit=5):
#     """Queries the pgvector database for the most relevant document chunks."""
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
        
#         # <=> calculates Cosine Distance. We subtract it from 1 to calculate Cosine Similarity.
#         cursor.execute("""
#             SELECT d.file_name, e.content, 1 - (e.embedding <=> %s::vector) as similarity
#             FROM document_elements e
#             JOIN production_documents d ON e.document_id = d.id
#             ORDER BY e.embedding <=> %s::vector
#             LIMIT %s;
#         """, (query_vector, query_vector, limit))
        
#         results = cursor.fetchall()
#         cursor.close()
#         conn.close()
#         return results
#     except Exception as e:
#         print(f"   ❌ [Error] Database query failed: {e}")
#         return []

def retrieve_context(user_question, query_vector, limit=10):
    """
    Executes a Hybrid Search (Semantic Vector + Full-Text Keyword Search)
    and merges rankings using Reciprocal Rank Fusion (RRF).
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Comprehensive Hybrid SQL Query using Common Table Expressions (CTEs)
        # We use plainto_tsquery to safely convert the raw query into tsquery format
        hybrid_sql = """
        WITH vector_search AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as rank
            FROM document_elements
            ORDER BY embedding <=> %s::vector
            LIMIT 20
        ),
        keyword_search AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC) as rank
            FROM document_elements
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            LIMIT 20
        )
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
        
        # Execute query with vector parameters and user question directly for plainto_tsquery
        cursor.execute(hybrid_sql, (query_vector, query_vector, user_question, user_question, limit))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return results
        
    except Exception as e:
        # Fallback to pure vector search if keyword parsing fails due to syntax characters
        print(f"   ⚠️ [Hybrid Search Notice] Text index failed or was empty, falling back to basic vector search... ({e})")
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

def assemble_prompt(user_question, search_results):
    """Formats the retrieved document contexts and constructs the strict LLM prompt."""
    context_blocks = []
    for row in search_results:
        file_name, content, similarity = row
        context_blocks.append(
            f"--- SOURCE DOCUMENT: {file_name} (Match Score: {similarity:.2f}) ---\n{content}\n"
        )
    
    full_context = "\n".join(context_blocks)
    
    prompt = f"""
    You are an expert, highly accurate healthcare insurance assistant. 
    Answer the user's question using ONLY the provided policy context below.
    If the answer is not contained in the context, say "I cannot find the answer in the provided documents."
    Always cite the Source Document name when providing facts. Do not hallucinate or make up information outside of the context.
    
    USER QUESTION: {user_question}
    
    CONTEXT:
    {full_context}
    """
    return prompt

def generate_answer(prompt):
    """Invokes the Gemini 2.5 Flash model to generate the grounded response."""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"   ❌ [Error] Text generation failed: {e}")
        return None

def ask_healthcare_bot(user_question):
    """Orchestrates the modular RAG execution pipeline."""
    # Step 1: Vectorize user query
    print(f"\n🧠 [1/4] Embedding query...")
    query_vector = get_google_embedding(user_question)
    if not query_vector:
        print("   ⚠️ Aborting search due to embedding failure.")
        return

    # Step 2: Retrieve relevant pages from postgres vault
    print(f"🔍 [2/4] Searching database vector store...")
    results = retrieve_context(user_question, query_vector)
    if not results:
        print("   ⚠️ No matching policies found in the database.")
        return

    # Step 3: Format the context and build prompt
    print(f"📝 [3/4] Fusing context and compiling prompt...")
    prompt = assemble_prompt(user_question, results)

    # Step 4: Call generation model
    print(f"💬 [4/4] Requesting response from model...")
    response_text = generate_answer(prompt)
    
    if response_text:
        print("\n========================= AI RESPONSE =========================")
        print(response_text)
        print("===============================================================")
    else:
        print("   ⚠️ Failed to get a response from the AI.")

def main():
    print("===============================================================")
    print(" 🏥 HEALTHCARE INSURANCE CO-PILOT (RAG TERMINAL ONLINE) ")
    print("===============================================================")
    try:
        while True:
            q = input("\nAsk a question about the policies (or type 'exit'): ")
            if q.strip().lower() == 'exit':
                print("\nShutting down terminal. Goodbye!")
                break
            if not q.strip():
                continue
            ask_healthcare_bot(q)
    except KeyboardInterrupt:
        print("\n\nShutting down terminal. Goodbye!")

if __name__ == "__main__":
    main()