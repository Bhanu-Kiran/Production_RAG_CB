import os
import psycopg2
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 1. INITIALIZATION & STATE MANAGEMENT
# ---------------------------------------------------------------------
load_dotenv()
client = genai.Client()

# Global chat memory storage: maintains tuples of (user_message, bot_response)
CHAT_HISTORY = []

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

# ---------------------------------------------------------------------
# 2. CONVERSATIONAL QUERY CONDENSATION
# ---------------------------------------------------------------------
def condense_query(new_question, history):
    """
    If the user asks a follow-up like "What is the limit?", this function 
    asks the LLM to rewrite it to "What is the room rent limit for Bajaj?" 
    so the PostgreSQL database actually knows what to search for.
    """
    if not history:
        return new_question

    history_context = ""
    for user_msg, bot_resp in history[-3:]:
        history_context += f"User: {user_msg}\nAssistant: {bot_resp[:150]}...\n"

    prompt = f"""
    Given the following conversation history and a brand new follow-up question, 
    rewrite the follow-up question into a standalone query that contains all necessary 
    context (such as company names or specific insurance terms mentioned earlier). 
    
    If the question is already fully independent and standalone, return it exactly as-is.
    Only return the final rewritten query string.

    HISTORY:
    {history_context}

    NEW QUESTION: {new_question}
    STANDALONE QUERY:
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"   ⚠️ [Condensation Notice] Core condensation failed, using raw query. ({e})")
        return new_question

# ---------------------------------------------------------------------
# 3. POSTGRESQL HYBRID SEARCH ENGINE
# ---------------------------------------------------------------------
def retrieve_hybrid_context(search_query, query_vector, limit=15):
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
        
        # Execute query. 
        # We pass query_vector twice for CTE 1. We pass search_query twice for CTE 2.
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

# ---------------------------------------------------------------------
# 4. ORCHESTRATION PIPELINE
# ---------------------------------------------------------------------
def assemble_prompt(user_question, search_results):
    """Formats the retrieved document contexts and constructs the strict LLM prompt."""
    context_blocks = []
    for row in search_results:
        file_name, content, similarity = row
        context_blocks.append(
            f"--- SOURCE DOCUMENT: {file_name} (Match Score: {similarity:.4f}) ---\n{content}\n"
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
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"   ❌ [Error] Text generation failed: {e}")
        return None

def ask_healthcare_bot(user_question):
    """Orchestrates the modular RAG execution pipeline."""
    global CHAT_HISTORY

    # Step 1: Condense the query based on past conversation history
    print(f"\n⚡ [1/5] Analyzing conversational context...")
    search_query = condense_query(user_question, CHAT_HISTORY)
    if search_query != user_question:
        print(f"   🔄 [Optimized Query]: \"{search_query}\"")
    else:
        print(f"   🎯 [Query Confirmed]: No rewriting required.")

    # Step 2: Vectorize user query
    print(f"🧠 [2/5] Embedding targeted search keys...")
    query_vector = get_google_embedding(search_query)
    if not query_vector:
        print("   ⚠️ Aborting search due to embedding failure.")
        return

    # Step 3: Retrieve relevant pages from postgres vault
    print(f"🔍 [3/5] Executing PostgreSQL Hybrid Search (Dense + Sparse)...")
    results = retrieve_hybrid_context(search_query, query_vector, limit=10)
    if not results:
        print("   ⚠️ No matching policies found in the database.")
        return

    # Step 4: Format the context and build prompt
    print(f"📝 [4/5] Fusing context and compiling prompt...")
    prompt = assemble_prompt(search_query, results)

    # Step 5: Call generation model
    print(f"💬 [5/5] Requesting response from model...")
    response_text = generate_answer(prompt)
    
    if response_text:
        print("\n========================= AI RESPONSE =========================")
        print(response_text)
        print("===============================================================")
        
        # Save interaction to global memory state
        CHAT_HISTORY.append((user_question, response_text))
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