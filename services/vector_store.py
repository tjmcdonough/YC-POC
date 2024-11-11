import chromadb
from typing import List, Dict

class VectorStoreService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.client.get_or_create_collection("documents")

    def add_documents(self, texts: List[str], metadata: List[Dict], ids: List[str]):
        self.collection.add(
            documents=texts,
            metadatas=metadata,
            ids=ids
        )

    def query_documents(self, query_text: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
