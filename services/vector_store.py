from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict
from dataclasses import dataclass
from langchain.load import dumps, loads


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

    def search(self, query_text: str, top_k=5) -> list[Document]:
        # Embed query text
        # embedding = self.embeddings.embed_query(query_text)

        # Perform similarity search
        results = self.vectorstore.similarity_search(query_text, k=top_k)

        return self.get_unique_union(results)

    def get_all_documents(self) -> List[Document]:
        """
        Retrieves all documents from the vector store.

        Returns:
            List[Document]: List of unique documents with their content and metadata

        Raises:
            Exception: If there's an error retrieving documents from the vector store
        """
        try:
            # Get all documents from the collection
            results = self.vectorstore._collection.get()

            if not results or not results['documents']:
                return []

            # Create Document objects from the raw results
            documents = [
                Document(page_content=doc,
                         metadata=meta if meta else {}) for doc, meta in zip(
                             results['documents'], results['metadatas'])
            ]

            # Return unique documents using the existing get_unique_union method
            return self.get_unique_union(documents)

        except Exception as e:
            raise Exception(
                f"Error retrieving documents from vector store: {str(e)}")

    def get_unique_union(self, documents: list[Document]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for doc in documents]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    def clear_data(self):
        try:
            # Use _collection.get() to get raw documents
            results = self.vectorstore._collection.get()
            if results and results['ids']:
                # Delete documents using the ids from results
                self.vectorstore._collection.delete(ids=results['ids'])
                # Persist the changes
                self.vectorstore.persist()
        except Exception as e:
            raise Exception(f"Error clearing vector store: {str(e)}")
