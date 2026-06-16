import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

sql_commands = """
DROP TABLE IF EXISTS document_elements CASCADE;
DROP TABLE IF EXISTS production_documents CASCADE;

CREATE TABLE production_documents (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) UNIQUE NOT NULL,
    clean_title VARCHAR(255) NOT NULL,
    global_summary TEXT NOT NULL,
    total_blocks INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_elements (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES production_documents(id) ON DELETE CASCADE,
    element_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    parent_element_id INT,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON document_elements USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX tsvector_idx ON document_elements USING gin(to_tsvector('english', content));
"""

try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()
    cursor.execute(sql_commands)
    conn.commit()
    print("🚀 Database migrated and tables created successfully!")
except Exception as e:
    print(f"❌ Migration failed: {e}")
finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals(): conn.close()