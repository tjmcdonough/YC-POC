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
                    processing_status TEXT DEFAULT 'processing',
                    total_chunks INTEGER DEFAULT 1,
                    processed_chunks INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add trigger for updated_at
            cur.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)
            
            cur.execute("""
                DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
                
                CREATE TRIGGER update_documents_updated_at
                    BEFORE UPDATE ON documents
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """)
        self.conn.commit()

    def save_document(self, filename: str, file_type: str, summary: str, metadata: Dict, total_chunks: int = 1) -> int:
        try:
            with self.conn.cursor() as cur:
                if isinstance(metadata, str):
                    metadata_json = metadata
                else:
                    metadata_json = json.dumps(metadata)
                
                cur.execute("""
                    INSERT INTO documents (
                        filename, file_type, summary, metadata, 
                        total_chunks, processing_status
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                    RETURNING id
                """, (
                    filename, file_type, summary, metadata_json,
                    total_chunks, 'processing' if total_chunks > 1 else 'completed'
                ))
                doc_id = cur.fetchone()
                if doc_id is None:
                    raise Exception("Failed to insert document")
                self.conn.commit()
                return doc_id[0]
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Error saving document: {str(e)}")

    def update_processing_status(self, doc_id: int, processed_chunks: int, status: str = None):
        try:
            with self.conn.cursor() as cur:
                if status is None:
                    # Auto-determine status based on processed chunks
                    cur.execute("""
                        UPDATE documents 
                        SET processed_chunks = %s,
                            processing_status = CASE 
                                WHEN %s >= total_chunks THEN 'completed'
                                ELSE 'processing'
                            END
                        WHERE id = %s
                    """, (processed_chunks, processed_chunks, doc_id))
                else:
                    cur.execute("""
                        UPDATE documents 
                        SET processed_chunks = %s,
                            processing_status = %s
                        WHERE id = %s
                    """, (processed_chunks, status, doc_id))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Error updating processing status: {str(e)}")

    def get_documents(self, filters: Optional[Dict] = None) -> List[Dict]:
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT 
                        id, filename, file_type, summary, metadata,
                        processing_status, total_chunks, processed_chunks,
                        created_at, updated_at
                    FROM documents
                """
                params = []
                
                if filters:
                    conditions = []
                    if 'file_type' in filters:
                        conditions.append("file_type = ANY(%s)")
                        params.append(filters['file_type'])
                    if 'date_range' in filters:
                        conditions.append("created_at BETWEEN %s AND %s")
                        params.extend([
                            filters['date_range']['start'],
                            filters['date_range']['end']
                        ])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC"
                
                cur.execute(query, params)
                return cur.fetchall() or []
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
