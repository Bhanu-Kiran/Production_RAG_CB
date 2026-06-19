import os
import time
import sys
import json
import psycopg2
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from pdf2image import convert_from_path, pdfinfo_from_path

# =====================================================================
# 1. INITIALIZATION & SETUP
# =====================================================================
print("🚀 Booting Fast Contextual Ingestion Engine...", flush=True)
load_dotenv(override=True)
POLICY_DIR = "./data/policies"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ CRITICAL ERROR: GEMINI_API_KEY missing from environment.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

def get_db_connection():
    """Constructs secure database connection from environment."""
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


# =====================================================================
# 2. CORE API OPERATIONS (FAST MEMORY STREAMING)
# =====================================================================
def extract_identity_fast(cover_image):
    """Extracts formal policy name and macro summary from Page 1."""
    prompt = """
    Analyze this insurance document cover page. Return a JSON object containing:
    1. "clean_title": A professional, cleaned, human-readable name of the policy.
    2. "global_summary": A 1-sentence macro description explaining what this document represents.
    Output ONLY raw, valid JSON.
    """
    try:
        # Pass the PIL Image directly in memory. Fast and efficient.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[cover_image, prompt]
        )
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data["clean_title"], data["global_summary"]
    except Exception as e:
        print(f"      [!] Identity Extraction Error: {e}")
        return "Unknown Policy", "Healthcare insurance policy document."

def extract_context_fast(page_image, clean_title, global_summary):
    """Generates the localized contextual summary and exact layout extraction."""
    prompt = f"""
    You are an expert healthcare insurance document extractor.
    Document Context: This page belongs to the '{clean_title}' policy. {global_summary}
    
    Task 1: Write a concise, 1-sentence description explaining exactly what rules, limits, definitions, or tables are displayed on this specific page.
    Task 2: Extract all narrative blocks and tables flawlessly. Keep tables in clean Markdown matrices.
    
    Format your output exactly as specified below with nothing else:
    [CONTEXT: Insert your 1-sentence localized description here]
    
    [RAW TEXT: Insert all extracted text and markdown tables here]
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[page_image, prompt]
        )
        return response.text.strip() if response and response.text else None
    except Exception as e:
        print(f"            [!] Vision Extraction Error: {e}")
        return None

def get_google_embedding_fast(text):
    """Generates a perfectly shaped 768-dimensional vector."""
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"            [!] Embedding Generation Error: {e}")
        return None


# =====================================================================
# 3. EXECUTIVE STREAMING ENGINE (WITH GRANULAR LOGGING)
# =====================================================================
def run_fast_ingestion():
    print("=====================================================================")
    print(" 🚀 UNLEASHING FAST CONTEXTUAL MULTI-MODAL PIPELINE")
    print("=====================================================================")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        pdf_files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        
        if total_files == 0:
            print(f"[!] Target directory '{POLICY_DIR}' is empty.")
            return

        for file_idx, file_name in enumerate(pdf_files, 1):
            file_path = os.path.join(POLICY_DIR, file_name)
            file_start_time = time.time()
            
            print(f"\n📁 [FILE {file_idx}/{total_files}]: {file_name}")
            
            # --- Document Registration ---
            info = pdfinfo_from_path(file_path)
            total_pages = info["Pages"]
            
            cursor.execute("SELECT id, clean_title, global_summary FROM production_documents WHERE file_name = %s;", (file_name,))
            doc_row = cursor.fetchone()
            
            if doc_row:
                doc_id, clean_title, global_summary = doc_row
                print(f"   └── [Log] Found registry ID: {doc_id} -> '{clean_title}'")
            else:
                print("   └── [Log] New document detected. Extracting identity from Cover Page...")
                # Render only page 1 for identity extraction
                cover_image = convert_from_path(file_path, dpi=100, first_page=1, last_page=1)[0]
                clean_title, global_summary = extract_identity_fast(cover_image)
                
                cursor.execute(
                    "INSERT INTO production_documents (file_name, clean_title, global_summary, total_blocks) VALUES (%s, %s, %s, %s) RETURNING id;", 
                    (file_name, clean_title, global_summary, total_pages)
                )
                doc_id = cursor.fetchone()[0]
                conn.commit()
                print(f"   └── [Log] Registered ID: {doc_id} -> '{clean_title}'")
            
            print(f"   └── [Log] Target length: {total_pages} pages. Launching high-speed processing loop...")
            
            # --- Page Processing Loop ---
            for page_num in range(1, total_pages + 1):
                # Idempotency Check
                page_signature = f"[CONTEXT: (Page {page_num})%"
                cursor.execute(
                    "SELECT id FROM document_elements WHERE document_id = %s AND content LIKE %s LIMIT 1;",
                    (doc_id, page_signature)
                )
                if cursor.fetchone():
                    print(f"        ⏭️ [Page {page_num}/{total_pages}] Found in Postgres. Skipping.")
                    continue
                
                print(f"        📄 [Page {page_num}/{total_pages}] Processing initiated...")
                
                # 1. Render Image
                render_start = time.time()
                # DPI 150 is the perfect balance between OCR clarity and memory speed
                images = convert_from_path(file_path, dpi=150, first_page=page_num, last_page=page_num)
                page_image = images[0]
                render_duration = time.time() - render_start
                print(f"            ├── [Image Rendered]: {render_duration:.2f}s")
                
                # 2. Vision API (Context Prepending + Extraction)
                gemini_start = time.time()
                raw_content = extract_context_fast(page_image, clean_title, global_summary)
                gemini_duration = time.time() - gemini_start
                
                if not raw_content:
                    print(f"            ├── [Vision Data]: Blank or non-extractable layout.")
                    raw_content = f"[CONTEXT: Blank Frame] [RAW TEXT: Page {page_num} layout empty.]"
                else:
                    print(f"            ├── [Vision Data]: Extraction complete in {gemini_duration:.2f}s")
                
                # Stamp absolute page number into context block
                final_content = raw_content.replace("[CONTEXT:", f"[CONTEXT: (Page {page_num})")
                
                # Show context snippet in terminal
                snippet = " ".join(final_content[:100].replace("\n", " ").split())
                print(f"            👁️  [Context Teaser]: \"{snippet}...\"")
                
                # 3. Vector Embeddings
                embed_start = time.time()
                vector = get_google_embedding_fast(final_content)
                embed_duration = time.time() - embed_start
                
                if vector:
                    print(f"            ├── [Vector Embed]: 768-D array calculated in {embed_duration:.2f}s")
                    
                    # 4. PostgreSQL Database Commit
                    db_start = time.time()
                    cursor.execute(
                        """
                        INSERT INTO document_elements 
                        (document_id, element_type, content, parent_element_id, embedding)
                        VALUES (%s, 'vision_contextual_page', %s, NULL, %s);
                        """,
                        (doc_id, final_content, vector)
                    )
                    conn.commit()
                    db_duration = time.time() - db_start
                    print(f"            └── [DB Commit]: Saved securely to Postgres in {db_duration:.4f}s")
                else:
                    print(f"            ❌ [Error]: Failed to generate vector. Page {page_num} omitted.")
                
            file_total_duration = time.time() - file_start_time
            print(f"   🏆 [File Complete]: Finished {file_name} in {file_total_duration:.2f} seconds!")
            
        print("\n=====================================================================")
        print(" 🎉 SUCCESS: FAST CONTEXTUAL PIPELINE COMPLETE.")
        print("=====================================================================")

    except Exception as e:
        print(f"\n[X] CRITICAL SYSTEM FAULT: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    try:
        run_fast_ingestion()
    except KeyboardInterrupt:
        print("\n\n🛑 Script interrupted by user.")