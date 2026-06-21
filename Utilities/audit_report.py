import os
import sys
import pandas as pd
# Add V3 directory to path so stage-specific imports work out-of-the-box
v3_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v3_hierarchical_summary_rag")
sys.path.append(v3_dir)
# Absolute production imports mapping to your V3 structure
from config.config import get_db_connection

def extract_summary(raw_content):
    """
    The Surgical Text Partitioner.
    Attempts to isolate the AI-generated summary prepended to the chunk.
    Falls back to a safe 500-character preview if no structural boundary is found.
    """
    if not raw_content:
        return "NO CONTENT"

    # Common structural boundaries used in AI contextual prepending
    delimiters = ["\n\n---", "RAW TEXT:", "\n\n", "Page Content:"]
    
    for delimiter in delimiters:
        if delimiter in raw_content:
            # Split the text at the first instance of the delimiter
            parts = raw_content.split(delimiter, 1)
            summary = parts[0].strip()
            
            # If the split accidentally just caught a header, grab a bit more
            if len(summary) > 20: 
                return summary

    # Fallback: If no clean delimiter is found, truncate to prevent Excel cell explosion
    return raw_content[:500] + "... [TRUNCATED FOR EXCEL PREVIEW]"

def generate_master_snapshot():
    print("\n" + "="*60)
    print(" 📊 GENERATING VECTOR DB MASTER SNAPSHOT (DUAL-SHEET)")
    print("="*60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # ---------------------------------------------------------
        # QUERY A: Fetch Document Metadata (Sheet 1)
        # ---------------------------------------------------------
        print("   ├─ Fetching Document Metadata...")
        cursor.execute("SELECT id, file_name, clean_title, created_at FROM production_documents;")
        docs_records = cursor.fetchall()
        
        # Convert to Pandas DataFrame
        df_docs = pd.DataFrame(docs_records, columns=['Document_ID', 'Original_File_Name', 'Clean_Title', 'Upload_Timestamp'])
        
        # ---------------------------------------------------------
        # QUERY B: Fetch Chunks & Human Foreign Key (Sheet 2)
        # ---------------------------------------------------------
        print("   ├─ Fetching Context Chunks (Omitting heavy vectors)...")
        # Notice we are only asking for 3 columns: id, clean_title, content
        chunk_query = """
            SELECT e.id, d.clean_title, e.content 
            FROM document_elements e
            JOIN production_documents d ON e.document_id = d.id
            ORDER BY d.clean_title;
        """
        cursor.execute(chunk_query)
        chunk_records = cursor.fetchall()
        
        # ---------------------------------------------------------
        # PHASE 3: Surgical Text Partitioning
        # ---------------------------------------------------------
        print("   ├─ Slicing AI Summaries from raw page data...")
        processed_chunks = []
        for row in chunk_records:
            # FIXED: We now safely map exactly to the 3 indices returned by SQL
            chunk_id = row[0]
            doc_name = row[1]
            raw_text = row[2] 
            
            # Apply our isolation logic
            clean_summary = extract_summary(raw_text)
            
            processed_chunks.append({
                'Chunk_ID': chunk_id,
                'Document_Name': doc_name,
                'AI_Context_Summary': clean_summary
            })
            
        df_chunks = pd.DataFrame(processed_chunks)

        # ---------------------------------------------------------
        # PHASE 4: Dual-Sheet Excel Generation
        # ---------------------------------------------------------
        export_filename = "Vector_DB_Master_Snapshot.xlsx"
        print(f"   ├─ Writing to multi-sheet workbook: {export_filename}...")
        
        # Using Pandas ExcelWriter to create distinct tabs
        with pd.ExcelWriter(export_filename, engine='openpyxl') as writer:
            df_docs.to_excel(writer, sheet_name='1_Documents_Metadata', index=False)
            df_chunks.to_excel(writer, sheet_name='2_Context_Summaries', index=False)
            
            # Auto-adjust column widths for readability 
            for sheet in writer.sheets.values():
                for column in sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter 
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    # Cap the maximum column width at 100 so it doesn't stretch infinitely
                    adjusted_width = min((max_length + 2), 100) 
                    sheet.column_dimensions[column_letter].width = adjusted_width

        print("   └─ ✅ EXPORT COMPLETE. File is ready for auditing.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ EXPORT FAILED: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    generate_master_snapshot()