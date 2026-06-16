# retriever.py
from config.config import client, get_db_connection

def get_embedding(text):
    """Generates a dense semantic vector representation."""
    from google.genai import types
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"   [!] Embedding step failed: {e}")
        return None

def execute_hybrid_search(search_query, target_companies, limit=5):
    """
    Executes a high-performance Reciprocal Rank Fusion (RRF) Hybrid Search
    against PostgreSQL with explicit, dynamic metadata filter isolation.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query_vector = get_embedding(search_query)
    if not query_vector:
        return []

    # Base building blocks for dynamic metadata WHERE clause execution
    where_clauses = ["de.element_type = 'vision_extracted_page'"]
    query_params = []

    # If the router isolated target companies, inject hard SQL isolation constraints
    if target_companies:
        company_filters = []
        for company in target_companies:
            company_filters.append("pd.clean_title ILIKE %s")
            query_params.append(f"%{company}%")
        where_clauses.append(f"({ ' OR '.join(company_filters) })")

    where_sql = "WHERE " + " AND ".join(where_clauses)

    # Core Reciprocal Rank Fusion SQL logic fusing Dense Vector + Sparse Keyword indices
    rrf_sql = f"""
    WITH dense_search AS (
        SELECT de.id, ROW_NUMBER() OVER (ORDER BY de.embedding <=> %s::vector) as rank
        FROM document_elements de
        JOIN production_documents pd ON de.document_id = pd.id
        {where_sql}
    ),
    sparse_search AS (
        SELECT de.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', de.content), plainto_tsquery('english', %s)) DESC) as rank
        FROM document_elements de
        JOIN production_documents pd ON de.document_id = pd.id
        {where_sql} AND to_tsvector('english', de.content) @@ plainto_tsquery('english', %s)
    )
    SELECT 
        de.content, 
        pd.clean_title, 
        COALESCE(1.0 / (60 + d.rank), 0.0) + COALESCE(1.0 / (60 + s.rank), 0.0) AS rrf_score
    FROM document_elements de
    JOIN production_documents pd ON de.document_id = pd.id
    LEFT JOIN dense_search d ON de.id = d.id
    LEFT JOIN sparse_search s ON de.id = s.id
    WHERE d.id IS NOT NULL OR s.id IS NOT NULL
    ORDER BY rrf_score DESC
    LIMIT %s;
    """

    # Assemble parameter sequencing dynamically
    # sequence: [vector, *where_params, sparse_text, *where_params, sparse_text, limit]
    full_params = [query_vector] + query_params + [search_query] + query_params + [search_query, limit]

    chunks = []
    try:
        cursor.execute(rrf_sql, full_params)
        rows = cursor.fetchall()
        for r in rows:
            chunks.append({
                "content": r[0],
                "source_policy": r[1],
                "score": float(r[2])
            })
    except Exception as e:
        print(f"   [!] Database execution exception: {e}")
    finally:
        cursor.close()
        conn.close()
        
    return chunks
