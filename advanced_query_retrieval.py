import psycopg2
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. CONFIGURATION & INFRASTRUCTURE SETUP
# ---------------------------------------------------------
# Define the dictionary to connect to our local Docker database
DB_CONFIG = {
    "dbname": "rag_prod_db",
    "user": "rag_admin",
    "password": "super_secure_password_123",
    "host": "localhost",
    "port": "5432"
}

def execute_hybrid_search(user_query, top_k_parents=2, rrf_k=60):
    """
    Executes a Hybrid Search (Vector + Keyword), fuses the ranks using RRF, 
    and returns the complete Parent Context blocks for the LLM.
    """
    
    # ---------------------------------------------------------
    # 2. LOCAL GPU EMBEDDING INITIALIZATION
    # ---------------------------------------------------------
    print(f"\n[System] Loading embedding model on RTX 4060 to process query: '{user_query}'")
    
    # Load the exact same Hugging Face model we used for ingestion directly into GPU memory
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    # Convert the user's text string into a 384-dimensional mathematical array
    query_vector = model.encode(user_query).tolist()
    
    # Open the communication pipeline to the PostgreSQL container
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # ---------------------------------------------------------
        # 3. SEARCH PATH A: DENSE VECTOR RETRIEVAL (Concepts & Synonyms)
        # ---------------------------------------------------------
        # We ask Postgres to calculate the Cosine Distance (<=>) between our query vector 
        # and every child chunk in the database. We pull the Top 50 closest matches.
        cursor.execute(
            """
            SELECT id, parent_id, child_text 
            FROM child_chunks 
            ORDER BY embedding <=> %s::vector ASC 
            LIMIT 50;
            """,
            (query_vector,)
        )
        vector_results = cursor.fetchall()
        
        # ---------------------------------------------------------
        # 4. SEARCH PATH B: SPARSE KEYWORD RETRIEVAL (Exact Text Matches)
        # ---------------------------------------------------------
        # We ask Postgres to tokenize our query into keywords (plainto_tsquery) and match 
        # it against the text index we built earlier. We pull the Top 50 highest frequency matches.
        cursor.execute(
            """
            SELECT id, parent_id, child_text 
            FROM child_chunks 
            WHERE to_tsvector('english', child_text) @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(to_tsvector('english', child_text), plainto_tsquery('english', %s)) DESC 
            LIMIT 50;
            """,
            (user_query, user_query)
        )
        keyword_results = cursor.fetchall()
        
        # ---------------------------------------------------------
        # 5. RECIPROCAL RANK FUSION (RRF) ALGORITHM
        # ---------------------------------------------------------
        # We create a dictionary to keep track of the combined RRF score for every unique chunk.
        rrf_scores = {}
        
        # We also need a dictionary to remember the Parent ID for each chunk so we can fetch it later.
        chunk_to_parent_map = {}
        
        # Process the Vector Search List: Loop through results and calculate RRF math based on their rank index.
        for rank, row in enumerate(vector_results):
            chunk_id = row[0]
            parent_id = row[1]
            
            # Map the chunk to its parent
            chunk_to_parent_map[chunk_id] = parent_id
            
            # Math: 1 / (60 + Rank Position). We add 1 to rank because enumerate starts at 0.
            score = 1.0 / (rrf_k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
            
        # Process the Keyword Search List: Do the exact same math and ADD it to the existing scores.
        for rank, row in enumerate(keyword_results):
            chunk_id = row[0]
            parent_id = row[1]
            
            chunk_to_parent_map[chunk_id] = parent_id
            
            # Math: Add keyword rank score to the vector rank score
            score = 1.0 / (rrf_k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
            
        # ---------------------------------------------------------
        # 6. SORTING & PARENT CONTEXT EXTRACTION
        # ---------------------------------------------------------
        # Sort the dictionary to find the chunks with the absolute highest combined RRF scores.
        sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # We want to pull the parent text for the top results, but we want to avoid duplicates.
        # If Child Chunk A and Child Chunk B both point to Parent 101, we only need Parent 101 once.
        unique_parent_ids = []
        for chunk_id, score in sorted_chunks:
            p_id = chunk_to_parent_map[chunk_id]
            if p_id not in unique_parent_ids:
                unique_parent_ids.append(p_id)
            
            # Stop collecting parents once we hit our requested budget (top_k_parents)
            if len(unique_parent_ids) >= top_k_parents:
                break
                
        # ---------------------------------------------------------
        # 7. FETCHING THE FINAL CONTEXT PAYLOAD
        # ---------------------------------------------------------
        final_contexts = []
        if unique_parent_ids:
            # We convert our list of parent IDs into a tuple format so SQL can read it
            format_strings = ','.join(['%s'] * len(unique_parent_ids))
            
            # We run a final, rapid lookup to grab the large parent paragraphs
            cursor.execute(
                f"SELECT id, parent_text FROM parent_contexts WHERE id IN ({format_strings});",
                tuple(unique_parent_ids)
            )
            parent_rows = cursor.fetchall()
            
            # Format the output into a clean list of text blocks
            for row in parent_rows:
                final_contexts.append(f"[PARENT CONTEXT ID: {row[0]}]\n{row[1]}")

        # Print the final result simulating what will be handed to the LLM
        print("\n" + "="*50)
        print("FINAL ASSEMBLED CONTEXT FOR LLM")
        print("="*50)
        for context in final_contexts:
            print(f"\n{context}\n" + "-"*50)
            
    except Exception as e:
        print(f"[X] Hybrid query crashed: {e}")
    finally:
        # Always securely close the database connections
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Test Scenario: We will ask a compound question requiring both concepts and exact keywords.
    test_query = "What is the policy for working from home, and where is Cluster-Omega-9 hosted?"
    execute_hybrid_search(test_query, top_k_parents=2)