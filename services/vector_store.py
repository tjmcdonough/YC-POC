from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict
from dataclasses import dataclass
from models.vector_document import VectorDocument


class VectorStoreService:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, chunk_overlap=50)
        self.vectorstore = Chroma(persist_directory="./chroma_store",
                                  embedding_function=self.embeddings)

    def add_documents(self, text: str, metadata: dict):

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        docs = [
            Document(page_content=chunk, metadata=metadata) for chunk in chunks
        ]

        # Index chunks in the vector store
        self.vectorstore.add_documents(docs)

    def search(self, query_text: str, top_k=5):
        # Embed query text
        embedding = self.embeddings.embed_query(query_text)

        # Perform similarity search
        results = self.vectorstore.similarity_search_by_vector(embedding,
                                                               k=top_k)
        return results
