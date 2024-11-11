from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict
from dataclasses import dataclass
import os

@dataclass
class SearchResult:
    documents: List[str]
    metadatas: List[Dict]
    distances: List[float]

class VectorStoreService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300,
            chunk_overlap=50
        )
        self.vectorstore = Chroma(
            persist_directory="./chroma_store",
            embedding_function=self.embeddings
        )
    
    def add_documents(self, texts: List[str], metadata: List[Dict], ids: List[str]):
        """
        Add documents to the vector store

        Args:
            texts: List of text content
            metadata: List of metadata dictionaries
            ids: List of unique identifiers
        """
        documents = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(texts, metadata)
        ]
        splits = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(documents=splits)
    
    def query_documents(self, query_text: str, n_results: int = 5) -> SearchResult:
        """
        Query documents using text similarity

        Args:
            query_text: Text to search for
            n_results: Number of results to return

        Returns:
            SearchResult object containing documents, metadata, and distances
        """
        docs = self.vectorstore.similarity_search(query_text, k=n_results)
        return SearchResult(
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
            distances=[1.0] * len(docs)  # Placeholder for compatibility
        )
