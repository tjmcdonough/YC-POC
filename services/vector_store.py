from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict
from dataclasses import dataclass

class VectorStoreService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=300, chunk_overlap=50)
            self.vectorstore = Chroma(persist_directory="./chroma_store",
                                    embedding_function=self.embeddings)
            self._initialized = True

    def add_documents(self, text: str, metadata: dict):
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        docs = [
            Document(page_content=chunk, metadata=metadata) for chunk in chunks
        ]

        # Index chunks in the vector store
        self.vectorstore.add_documents(docs)
        # Persist after adding documents
        self.vectorstore.persist()

    def search(self, query_text: str, top_k=5):
        # Embed query text
        embedding = self.embeddings.embed_query(query_text)

        # Perform similarity search
        results = self.vectorstore.similarity_search_by_vector(embedding,
                                                             k=top_k)
        return results

    def clear_data(self):
        try:
            # Get all document IDs
            ids = [doc.id for doc in self.vectorstore.get()]
            if ids:
                # Delete documents by IDs
                self.vectorstore._collection.delete(ids=ids)
                # Persist the changes
                self.vectorstore.persist()
        except Exception as e:
            raise Exception(f"Error clearing vector store: {str(e)}")
