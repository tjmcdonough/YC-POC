import os
import psycopg2
from psycopg2.extras import RealDictCursor
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.conn.commit()

    def save_document(self, filename: str, file_type: str, summary: str, metadata: Dict) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (filename, file_type, summary, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (filename, file_type, summary, metadata))
            doc_id = cur.fetchone()[0]
        self.conn.commit()
        return doc_id

    def get_documents(self, query: Optional[Dict] = None) -> List[Dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if query:
                cur.execute("""
                    SELECT * FROM documents 
                    WHERE metadata @> %s::jsonb
                    ORDER BY created_at DESC
                """, (query,))
            else:
                cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return cur.fetchall()
