import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import Dict, List, Optional

class DatabaseService:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            host=os.environ["PGHOST"],
            port=os.environ["PGPORT"]
        )
        self._create_tables()

    def _create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    summary TEXT,
                    metadata JSONB,
                    processing_status TEXT DEFAULT 'completed',
                    total_chunks INTEGER DEFAULT 1,
                    processed_chunks INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.conn.commit()

    def save_document(self, filename: str, file_type: str, summary: str, metadata: Dict, total_chunks: int = 1) -> int:
        try:
            with self.conn.cursor() as cur:
                # Ensure metadata is properly serialized
                if isinstance(metadata, str):
                    metadata_json = metadata  # Already a JSON string
                else:
                    metadata_json = json.dumps(metadata)
                
                cur.execute("""
                    INSERT INTO documents (filename, file_type, summary, metadata, total_chunks, processing_status)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                    RETURNING id
                """, (filename, file_type, summary, metadata_json, total_chunks, 'processing' if total_chunks > 1 else 'completed'))
                doc_id = cur.fetchone()
                if doc_id is None:
                    raise Exception("Failed to insert document")
                self.conn.commit()
                return doc_id[0]
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Error saving document: {str(e)}")

    def update_processing_status(self, doc_id: int, processed_chunks: int, status: str = 'processing'):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents 
                    SET processed_chunks = %s, 
                        processing_status = %s,
                        summary = CASE 
                            WHEN %s = 'completed' AND summary = 'Processing...'
                            THEN NULL
                            ELSE summary
                        END
                    WHERE id = %s
                """, (processed_chunks, status, status, doc_id))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Error updating processing status: {str(e)}")

    def get_documents(self, query: Optional[Dict] = None) -> List[Dict]:
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                if query:
                    cur.execute("""
                        SELECT * FROM documents 
                        WHERE metadata @> %s::jsonb
                        ORDER BY created_at DESC
                    """, (json.dumps(query),))
                else:
                    cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
                return cur.fetchall() or []
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
