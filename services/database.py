import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import json
from typing import Dict, List, Optional
import time
import logging

class DatabaseService:
    def __init__(self, min_connections=1, max_connections=10, max_retries=3):
        self.max_retries = max_retries
        self.pool = ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            dbname=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            host=os.environ["PGHOST"],
            port=os.environ["PGPORT"]
        )
        self._create_tables()

    def __del__(self):
        """Ensure pool is closed when service is destroyed"""
        if hasattr(self, 'pool'):
            self.pool.closeall()

    @contextmanager
    def get_connection(self):
        """Context manager for safely acquiring and releasing connections"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn is not None:
                self.pool.putconn(conn)

    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute database operation with retry mechanism"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with self.get_connection() as conn:
                    with conn.cursor(cursor_factory=kwargs.get('cursor_factory', None)) as cur:
                        cur.execute(*args)
                        if operation == 'fetch_one':
                            result = cur.fetchone()
                        elif operation == 'fetch_all':
                            result = cur.fetchall()
                        else:
                            result = None
                    conn.commit()
                    return result
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                last_error = e
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue
            except Exception as e:
                conn.rollback() if conn else None
                raise Exception(f"Database error: {str(e)}")
        raise Exception(f"Max retries exceeded. Last error: {str(last_error)}")

    def _create_tables(self):
        """Create necessary database tables"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create documents table
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
                    
                    # Create or replace the updated_at function
                    cur.execute('''
                        CREATE OR REPLACE FUNCTION update_updated_at_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = CURRENT_TIMESTAMP;
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                    ''')

                    # Drop existing trigger if exists and create new one
                    cur.execute('''
                        DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
                    ''')

                    cur.execute('''
                        CREATE TRIGGER update_documents_updated_at
                        BEFORE UPDATE ON documents
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                    ''')

                    # Ensure the updated_at column exists
                    cur.execute('''
                        DO $$ 
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 
                                FROM information_schema.columns 
                                WHERE table_name = 'documents' 
                                AND column_name = 'updated_at'
                            ) THEN
                                ALTER TABLE documents 
                                ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                            END IF;
                        END $$;
                    ''')
                conn.commit()
        except Exception as e:
            raise Exception(f"Error creating tables: {str(e)}")

    def save_document(self, filename: str, file_type: str, summary: str, metadata: Dict, total_chunks: int = 1) -> int:
        """Save document with retry mechanism"""
        try:
            metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata
            result = self._execute_with_retry(
                'fetch_one',
                """
                INSERT INTO documents (
                    filename, file_type, summary, metadata, 
                    total_chunks, processing_status
                )
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """,
                (filename, file_type, summary, metadata_json,
                 total_chunks, 'processing' if total_chunks > 1 else 'completed')
            )
            
            if result is None:
                raise Exception("Failed to insert document")
            return result[0]
        except Exception as e:
            raise Exception(f"Error saving document: {str(e)}")

    def update_processing_status(self, doc_id: int, processed_chunks: int, status: str = None):
        """Update document processing status with retry mechanism"""
        try:
            if status is None:
                self._execute_with_retry(
                    'execute',
                    """
                    UPDATE documents 
                    SET processed_chunks = %s,
                        processing_status = CASE 
                            WHEN %s >= total_chunks THEN 'completed'
                            ELSE 'processing'
                        END
                    WHERE id = %s
                    """,
                    (processed_chunks, processed_chunks, doc_id)
                )
            else:
                self._execute_with_retry(
                    'execute',
                    """
                    UPDATE documents 
                    SET processed_chunks = %s,
                        processing_status = %s
                    WHERE id = %s
                    """,
                    (processed_chunks, status, doc_id)
                )
        except Exception as e:
            raise Exception(f"Error updating processing status: {str(e)}")

    def get_documents(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Get documents with retry mechanism"""
        try:
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
            
            return self._execute_with_retry('fetch_all', query, params, cursor_factory=RealDictCursor) or []
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
