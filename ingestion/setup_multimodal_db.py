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

# Initialize the official Google GenAI Client with a strict 60-second timeout 
# to prevent silent network freezes if a packet drops.
client = genai.Client(http_options={'timeout': 60.0})

def get_db_connection():
    """Establishes the connection to the PostgreSQL vector database."""
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
        
        # If Gemini successfully processes the page but finds no text (like a logo page)
        if response.text:
            return response.text.strip()
        return ""  # Return empty string for genuine blank pages

    except Exception as e:
        # Return None ONLY if a network crash, timeout, or API failure occurs
        print(f"      [!] Vision Extraction Error on Page {page_num}: {e}")
        return None

def get_google_embedding(text):
    """Generates a 768-dimensional semantic vector via Paid Tier API."""
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
                
                # Check database to see if this page was already successfully processed
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
                
                # --- SAFETY LOGIC FOR NETWORK DROPS VS BLANK PAGES ---
                if page_content is None:
                    print(f"            ❌ [Network Timeout]: Connection failed. Skipping DB save to preserve checkpoint.")
                    time.sleep(2) # Brief pause before trying the next page
                    continue