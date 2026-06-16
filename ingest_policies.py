import os
import psycopg2
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm

# 1. ENVIRONMENT & BOOTSTRAPING
load_dotenv()

POLICY_DIR = "./data/policies"
IMAGE_OUTPUT_DIR = "./data/extracted_images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# Instantiate the Google GenAI client (picks up GEMINI_API_KEY automatically from environment)
client = genai.Client()

def get_db_connection():
    """Constructs database connections securely using system environment variables."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def get_google_embedding(text):
    """Generates a native 768-D embedding from Google's new gemini-embedding-2 model."""
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            # We use Matryoshka Representation to shrink the 3072 vector down to 768 to fit our database
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"\n[!] Embedding Engine Error: {e}")
        return None

def process_visual_diagram(image_path):
    """Sends cropped images/flowcharts to Gemini 2.5 Flash for continuous description mapping."""
    try:
        uploaded_file = client.files.upload(file=image_path)
        prompt = """
        You are an expert healthcare document analyst. Decipher this visual flowchart, 
        diagram, or table layout. Translate all nodes, text arrows, decision trees, 
        and values into a detailed, dense text description for a semantic search engine.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[uploaded_file, prompt]
        )
        return response.text.strip()
    except Exception as e:
        return f"[VISUAL EXTRACTION COMPROMISED: {e}]"

def run_multimodal_ingestion():
    print("==================================================================")
    print(" EXECUTION PROTOCOL: MULTIMODAL INGESTION OVER THE 12 TARGET DOCS" )
    print("==================================================================")
    
    # Open infrastructure connection
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Scan directory for target PDFs
        pdf_files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        
        if total_files == 0:
            print(f"[X] Directory Verification Error: No PDFs found inside {POLICY_DIR}")
            return

        print(f"[System] Ingestion pipeline target locked on {total_files} active files.")
        
        # Core Document Extraction Iteration
        for file_idx, file_name in enumerate(pdf_files, 1):
            file_path = os.path.join(POLICY_DIR, file_name)
            print(f"\n---> [Processing PDF {file_idx}/{total_files}]: {file_name}")
            
            # Step A: Register the document tracking row
            cursor.execute(
                "INSERT INTO production_documents (file_name) VALUES (%s) RETURNING id;", 
                (file_name,)
            )
            print("     [+] Registered document in 'production_documents' table.")
            doc_id = cursor.fetchone()[0]
            
            # Step B: Vision/Layout Segmentation (Natively offloads to RTX 4060 CUDA if active)
            elements = partition_pdf(
                filename=file_path,
                strategy="fast",
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=False,
                extract_image_block_output_dir=IMAGE_OUTPUT_DIR,
                chunking_strategy="by_title" # Ties deep indentation back to sections
            )
            
            total_elements = len(elements)
            print(f"     [+] Dissected document into {total_elements} visual components.")
            
            # The context memory cell used to map parent-child indentation links
            current_parent_header_id = None
            
            # Step C: Element Routing, Embedding Generation, and Storage
            print(f"     [+] Processing components...")
            for element in tqdm(elements, desc=f"         {file_name[:20]}...", unit="elem"):
                element_type = type(element).__name__
                payload_content = None
                db_element_type = "text"
                
                # Image Pipeline Routing
                if element_type == "Image":
                    image_path = element.metadata.image_path
                    if image_path and os.path.exists(image_path):
                        payload_content = process_visual_diagram(image_path)
                        db_element_type = "image_description"
                
                # Table Pipeline Routing
                elif element_type == "Table":
                    payload_content = element.metadata.text_as_html or str(element)
                    db_element_type = "table_markdown"
                
                # Block Text / Indentation Handling
                elif element_type in ["CompositeElement", "NarrativeText", "Title", "ListItem"]:
                    payload_content = str(element)
                    db_element_type = "text"
                
                if not payload_content or not payload_content.strip():
                    continue
                
                # Vectorization Stage via Google API
                vector = get_google_embedding(payload_content)
                
                if vector:
                    cursor.execute(
                        """
                        INSERT INTO document_elements 
                        (document_id, element_type, content, parent_element_id, embedding)
                        VALUES (%s, %s, %s, %s, %s) RETURNING id;
                        """,
                        (doc_id, db_element_type, payload_content, current_parent_header_id, vector)
                    )
                    inserted_row_id = cursor.fetchone()[0]
                    
                    # If this block acts as a Title/Section-Header, track its ID
                    # Upcoming sub-bullets or child paragraphs will be linked to it
                    if element_type == "Title":
                        current_parent_header_id = inserted_row_id
            
            # Close out metrics for this file
            cursor.execute(
                "UPDATE production_documents SET total_blocks = %s WHERE id = %s;",
                (total_elements, doc_id)
            )
            conn.commit()
            print(f"     [✓] Ingestion sequence fully synchronized for: {file_name}")

        print("\n==================================================================")
        print(f" [✓] PIPELINE COMPLETION PROTOCOL: ALL {total_files} DOCUMENTS COMMITTED")
        print("==================================================================")

    except Exception as e:
        print(f"\n[X] CRITICAL STRUCTURAL CRASH IN PIPELINE: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_multimodal_ingestion()