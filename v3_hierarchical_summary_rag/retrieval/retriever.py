import sys
from pathlib import Path

# Add parent directory to sys.path to allow imports from config
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import CrossEncoder
from config.config import client, get_db_connection
from google.genai import types

# ------------------------------------------------------------------------
# LOCAL MACHINE LEARNING MODELS INITIALIZATION
# ------------------------------------------------------------------------
print("\n🧠 Initializing Local Phase 3 Cross-Encoder ('ms-marco-MiniLM-L-6-v2')...")
print("   (Runs fully offline using local CPU/RAM context windows)")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
print("✅ Local Cross-Encoder loaded into memory successfully.\n")

def get_query_embedding(text):
    """Generates 768-D query vector via Gemini Embedding API."""
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"            [!] Retrieval Embedding Error: {e}")
        return None

def apply_mmr(doc_embeddings, query_embedding, docs, top_k=20, diversity_penalty=0.4):
    """
    Executes Maximal Marginal Relevance (MMR) locally via NumPy matrix math.
    Filters out information redundancy.
    """
    if not docs:
        return []
        
    selected_indices = []
    unselected_indices = list(range(len(docs)))
    
    query_similarities = [1 - cosine(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    absolute_best_match_idx = np.argmax(query_similarities)
    selected_indices.append(absolute_best_match_idx)
    unselected_indices.remove(absolute_best_match_idx)
    
    while len(selected_indices) < top_k and unselected_indices:
        best_mmr_score = -np.inf
        candidate_idx_to_select = -1
        
        for candidate_idx in unselected_indices:
            semantic_relevance = query_similarities[candidate_idx]
            max_redundancy = max(
                [1 - cosine(doc_embeddings[candidate_idx], doc_embeddings[already_selected_idx]) 
                 for already_selected_idx in selected_indices]
            )
            
            mmr_score = (1 - diversity_penalty) * semantic_relevance - diversity_penalty * max_redundancy
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                candidate_idx_to_select = candidate_idx
                
        selected_indices.append(candidate_idx_to_select)
        unselected_indices.remove(candidate_idx_to_select)
        
    return [docs[i] for i in selected_indices]

def rerank_via_local_model(query, chunks):
    """
    Scores query-document text pairs locally with full deep contextual attention.
    Guarantees structural and logical alignment over basic vector proximity.
    """
    if not chunks:
        return []
        
    pairs = [[query, chunk["content"]] for chunk in chunks]
    cross_scores = reranker.predict(pairs)
    
    for idx, score in enumerate(cross_scores):
        chunk = chunks[idx]
        chunk["cross_score"] = float(score)
        
    # Sort descending by local cross-encoder score
    return sorted(chunks, key=lambda x: x.get("cross_score", -100.0), reverse=True)

def execute_single_search_pipeline(query_string, doc_ids, over_fetch_limit, mmr_k, cursor):
    """Coordinates the local 3-tier filtration process for an individual query string."""
    query_vector = get_query_embedding(query_string)
    if not query_vector:
        return []

    # 1. Local PGVector Over-Fetch (Targeting 50 documents)
    if doc_ids:
        sql_query = """
            SELECT e.content, e.embedding, d.clean_title
            FROM document_elements e
            JOIN production_documents d ON e.document_id = d.id
            WHERE e.document_id = ANY(%s)
            ORDER BY (e.embedding <=> %s::vector) ASC
            LIMIT %s;
        """
        cursor.execute(sql_query, (query_vector, doc_ids, over_fetch_limit))
    else:
        sql_query = """
            SELECT e.content, e.embedding, d.clean_title
            FROM document_elements e
            JOIN production_documents d ON e.document_id = d.id
            ORDER BY (e.embedding <=> %s::vector) ASC
            LIMIT %s;
        """
        cursor.execute(sql_query, (query_vector, over_fetch_limit))

    records = cursor.fetchall()
    
    raw_candidate_docs = []
    extracted_doc_embeddings = []
    
    for rec in records:
        if rec[0] not in [d["content"] for d in raw_candidate_docs]:
            raw_candidate_docs.append({"content": rec[0], "source_document": rec[2]})
            numpy_vector_array = np.array([float(x) for x in rec[1].strip('[]').split(',')])
            extracted_doc_embeddings.append(numpy_vector_array)
    
    print(f"      [Stage 1: Over-Fetch] Vector DB retrieved {len(raw_candidate_docs)} candidate chunks.")

    # 2. Local MMR Filtering (Compresses initial 50 candidates down to 20 unique items)
    diverse_chunks = apply_mmr(extracted_doc_embeddings, query_vector, raw_candidate_docs, top_k=mmr_k)
    print(f"      [Stage 2: MMR Selection] Filtered down to {len(diverse_chunks)} highly unique chunks.")
    
    # 3. Local Cross-Encoder Scoring (Deep transformer verification)
    reranked_chunks = rerank_via_local_model(query_string, diverse_chunks)
    return reranked_chunks
def execute_hybrid_retrieval(routing_payload, over_fetch_limit=50, mmr_k=20, final_k=6):
    intent = routing_payload.get("intent", "single_search")
    target_docs = routing_payload.get("target_documents", [])
    search_queries = routing_payload.get("search_queries", [])
    
    if intent == "chitchat" or not search_queries:
        return []

    # --- SAFETY TWEAK 1: Dynamic CPU Scaling ---
    # If there are multiple queries, reduce the number of chunks sent to the heavy Cross-Encoder
    # to prevent 5-minute CPU hangs.
    dynamic_mmr_k = max(5, mmr_k // len(search_queries)) if search_queries else mmr_k

    conn = get_db_connection()
    cursor = conn.cursor()
    aggregated_results = []

    try:
        doc_ids = []
        if target_docs:
            for doc_name in target_docs:
                cursor.execute(
                    "SELECT id FROM production_documents WHERE clean_title ILIKE %s OR file_name ILIKE %s;",
                    (f"%{doc_name}%", f"%{doc_name}%")
                )
                rows = cursor.fetchall()
                for r in rows:
                    doc_ids.append(r[0])
            
            # --- SAFETY TWEAK 2: SQL Fallback ---
            # If the user typed a heavily misspelled company name that SQL couldn't find,
            # drop the strict filter and search globally instead of returning 0 chunks.
            if not doc_ids:
                print(f"   ⚠️ [Metadata Shield] No exact DB match for {target_docs}. Reverting to global search.")

        for original_query in search_queries:
            print(f"   🔍 Query Execution: '{original_query}'")
            # Notice we pass dynamic_mmr_k here instead of mmr_k
            reranked_chunks = execute_single_search_pipeline(
                query_string=original_query, 
                doc_ids=doc_ids, 
                over_fetch_limit=over_fetch_limit, 
                mmr_k=dynamic_mmr_k, 
                cursor=cursor
            )
            
            for chunk in reranked_chunks[:final_k]:
                if chunk["content"] not in [r["content"] for r in aggregated_results]:
                    aggregated_results.append(chunk)

        print(f"   📊 [Final Context Size] Yielding top {len(aggregated_results[:final_k])} validated context pages.")
        return aggregated_results[:final_k]

    except Exception as e:
        print(f"❌ RETRIEVAL ENGINE FAULT: {e}")
        return []
    finally:
        cursor.close()
        conn.close()
    """
    Phased Retrieval Orchestrator (Phase 3 Core).
    Funnels inputs through local DB search, local MMR, and local Cross-Encoder reranking.
    """
    intent = routing_payload.get("intent", "single_search")
    target_docs = routing_payload.get("target_documents", [])
    search_queries = routing_payload.get("search_queries", [])
    
    if intent == "chitchat" or not search_queries:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()
    aggregated_results = []

    try:
        # Resolve any target metadata boundaries passed down by the router
        doc_ids = []
        if target_docs:
            for doc_name in target_docs:
                cursor.execute(
                    "SELECT id FROM production_documents WHERE clean_title ILIKE %s OR file_name ILIKE %s;",
                    (f"%{doc_name}%", f"%{doc_name}%")
                )
                rows = cursor.fetchall()
                for r in rows:
                    doc_ids.append(r[0])

        # Loop through generated sub-queries
        for original_query in search_queries:
            print(f"   🔍 Query Execution: '{original_query}'")
            reranked_chunks = execute_single_search_pipeline(
                query_string=original_query, 
                doc_ids=doc_ids, 
                over_fetch_limit=over_fetch_limit, 
                mmr_k=mmr_k, 
                cursor=cursor
            )
            
            # Extract the top slices following local verification and deduplicate them
            for chunk in reranked_chunks[:final_k]:
                if chunk["content"] not in [r["content"] for r in aggregated_results]:
                    aggregated_results.append(chunk)

        # Print final extraction metrics for terminal logging transparency
        print(f"   📊 [Final Context Size] Yielding top {len(aggregated_results[:final_k])} validated context pages.")
        return aggregated_results[:final_k]

    except Exception as e:
        print(f"❌ RETRIEVAL ENGINE FAULT: {e}")
        return []
    finally:
        cursor.close()
        conn.close()