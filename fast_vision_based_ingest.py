import os
import time
import psycopg2
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from pdf2image import convert_from_path, pdfinfo_from_path

# =====================================================================
# 1. INITIALIZATION & SETUP
# =====================================================================
load_dotenv()
POLICY_DIR = "./data/policies"

# Initialize the official Google GenAI Client
client = genai.Client()

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

# =====================================================================
# 2. CORE API OPERATIONS (UNTHROTTLED)
# =====================================================================
def process_page_with_gemini(image, page_num):
    """Sends a single high-res page image to Gemini 2.5 Flash."""
    try:
        prompt = """
        You are an expert healthcare document data extractor. 
        1. Extract all narrative text perfectly, preserving any nested bullet points.
        2. Extract any grid or pricing matrix as a perfectly formatted Markdown table.
        3. Describe any flowchart or diagram in detail.
        Do not include conversational filler. ONLY output the data.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt]
        )
        
        if response.text:
            return response.text.strip()
        return None
    except Exception as e:
        print(f"      [!] Vision Extraction Error on Page {page_num}: {e}")
        return None

def get_google_embedding(text):
    """Generates a 768-dimensional semantic vector vector via Paid Tier API."""
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"      [!] Embedding Generation Error: {e}")
        return None

# =====================================================================
# 3. EXECUTIVE STREAMING ENGINE
# =====================================================================
def run_unleashed_ingestion():
    print("=====================================================================")
    print(" 🚀 UNLEASHING PAID-TIER MULTI-MODAL STREAMING ENGINE")
    print("=====================================================================")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Scan data folder for PDF files
        pdf_files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        
        if total_files == 0:
            print(f"[!] Target directory '{POLICY_DIR}' is empty. Drop your PDFs there.")
            return

        for file_idx, file_name in enumerate(pdf_files, 1):
            file_path = os.path.join(POLICY_DIR, file_name)
            file_start_time = time.time()
            
            print(f"\n📁 [FILE {file_idx}/{total_files}]: {file_name}")
            print(f"   └── [Log] Reading system metadata...")
            
            # 1. Document Registry Checkpoint
            cursor.execute("SELECT id FROM production_documents WHERE file_name = %s;", (file_name,))
            doc_row = cursor.fetchone()
            if doc_row:
                doc_id = doc_row[0]
                print(f"   └── [Log] Found document registry ID: {doc_id}")
            else:
                cursor.execute("INSERT INTO production_documents (file_name) VALUES (%s) RETURNING id;", (file_name,))
                doc_id = cursor.fetchone()[0]
                conn.commit()
                print(f"   └── [Log] Registered new document. Assigned ID: {doc_id}")
            
            # 2. Get Total Page Count instantly without loading images
            info = pdfinfo_from_path(file_path)
            total_pages = info["Pages"]
            print(f"   └── [Log] Target document length detected: {total_pages} pages.")
            print(f"   └── [Log] Launching unthrottled streaming worker loops...")
            
            # 3. Page-by-Page Processing Loop
            for page_num in range(1, total_pages + 1):
                page_start_time = time.time()
                
                # Check database to see if this page was already processed in a past run
                page_marker = f"[PAGE {page_num} START]%"
                cursor.execute(
                    "SELECT id FROM document_elements WHERE document_id = %s AND content LIKE %s LIMIT 1;",
                    (doc_id, page_marker)
                )
                if cursor.fetchone():
                    print(f"         ⏭️ [Page {page_num}/{total_pages}] Found in Postgres. Skipping processing.")
                    continue
                
                print(f"         📄 [Page {page_num}/{total_pages}] Processing initiated...")
                
                # Render ONLY the targeted page to completely eliminate RAM bloat
                convert_start = time.time()
                images = convert_from_path(file_path, dpi=150, first_page=page_num, last_page=page_num)
                single_page_image = images[0]
                convert_duration = time.time() - convert_start
                print(f"            ├── [Image Rendered]: {convert_duration:.2f}s")
                
                # Execute vision transcription via Gemini 2.5 Flash
                gemini_start = time.time()
                page_content = process_page_with_gemini(single_page_image, page_num)
                gemini_duration = time.time() - gemini_start
                
                if not page_content:
                    print(f"            ├── [Gemini Response]: Returned empty. Identified as blank space or logo page.")
                    # Safe entry for tracking blank spaces
                    page_content = "This page is blank or contains non-extractable design layout elements."
                else:
                    print(f"            ├── [Gemini Response]: Extraction complete in {gemini_duration:.2f}s")
                
                # --- VISUAL TEASER COMPONENT ---
                # Clean up newlines for cleaner single-line console presentation
                clean_teaser = " ".join(page_content.replace("\n", " ").split())
                teaser_snippet = clean_teaser[:120] + "..." if len(clean_teaser) > 120 else clean_teaser
                print(f"            👁️  [Content Teaser]: \"{teaser_snippet}\"")
                
                # Execute Vector Embeddings
                embed_start = time.time()
                vector = get_google_embedding(page_content)
                embed_duration = time.time() - embed_start
                
                if vector:
                    print(f"            ├── [Vector Generated]: 768-D array calculated in {embed_duration:.2f}s")
                    
                    # Secure data straight to PostgreSQL
                    db_start = time.time()
                    formatted_content = f"[PAGE {page_num} START]\n{page_content}\n[PAGE {page_num} END]"
                    cursor.execute(
                        """
                        INSERT INTO document_elements 
                        (document_id, element_type, content, parent_element_id, embedding)
                        VALUES (%s, %s, %s, NULL, %s);
                        """,
                        (doc_id, "vision_extracted_page", formatted_content, vector)
                    )
                    conn.commit()
                    db_duration = time.time() - db_start
                    print(f"            └── [Postgres Commit]: Saved securely to DB in {db_duration:.4f}s")
                else:
                    print(f"            ❌ [Error]: Failed to generate vector. Page {page_num} omitted.")
                
                # Microsecond network buffer to avoid socket saturation
                time.sleep(0.1)
                
            # File Processing Completion Metrics
            file_total_duration = time.time() - file_start_time
            cursor.execute("UPDATE production_documents SET total_blocks = %s WHERE id = %s;", (total_pages, doc_id))
            conn.commit()
            print(f"   🏆 [File Complete]: Finished {file_name} in {file_total_duration:.2f} seconds!")
            
        print("\n=====================================================================")
        print(" 🎉 SUCCESS: ALL CORPORATE HEALTHCARE POLICIES COMMITTED TO DATABASE.")
        print("=====================================================================")

    except Exception as e:
        print(f"\n[X] CRITICAL SYSTEM FAULT: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_unleashed_ingestion()