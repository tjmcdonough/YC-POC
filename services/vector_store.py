import os
import weaviate
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class SearchResult:
    documents: List[str]
    metadatas: List[Dict]
    distances: List[float]


class VectorStoreService:

    def __init__(self):
        self.client = weaviate.Client(
            url=os.environ["WEAVIATE_URL"],
            auth_client_secret=weaviate.AuthApiKey(
                api_key=os.environ["WEAVIATE_API_KEY"]))

        # Define the class schema if it doesn't exist
        self._create_schema()

    def _create_schema(self):
        """Create the schema for the Document class if it doesn't exist"""
        schema = {
            "class":
            "Document",
            "vectorizer":
            "text2vec-transformers",  # Default text vectorizer
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": False
                }
            },
            "properties": [{
                "name": "content",
                "dataType": ["text"],
                "description": "The main text content of the document",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": False,
                        "vectorizePropertyName": False
                    }
                }
            }, {
                "name":
                "metadata",
                "dataType": ["object"],
                "description":
                "Additional metadata for the document"
            }]
        }

        # Check if schema exists, if not create it
        try:
            self.client.schema.get()["classes"]
        except:
            self.client.schema.create_class(schema)

    def add_documents(self, texts: List[str], metadata: List[Dict],
                      ids: List[str]):
        """
        Add documents to the vector store

        Args:
            texts: List of text content
            metadata: List of metadata dictionaries
            ids: List of unique identifiers
        """
        batch = weaviate.batch.ObjectsBatcher(client=self.client,
                                              batch_size=100)

        for text, meta, doc_id in zip(texts, metadata, ids):
            properties = {"content": text, "metadata": meta}

            batch.add_data_object(data_object=properties,
                                  class_name="Document",
                                  uuid=doc_id)

        batch.flush()

    def query_documents(self,
                        query_text: str,
                        n_results: int = 5) -> SearchResult:
        """
        Query documents using text similarity

        Args:
            query_text: Text to search for
            n_results: Number of results to return

        Returns:
            SearchResult object containing documents, metadata, and distances
        """
        query = (self.client.query.get(
            "Document", ["content", "metadata"]).with_near_text({
                "concepts": [query_text]
            }).with_limit(n_results).with_additional(["distance"]))

        results = query.do()

        # Extract results
        documents = []
        metadatas = []
        distances = []

        if "data" in results and "Get" in results["data"]:
            for item in results["data"]["Get"]["Document"]:
                documents.append(item["content"])
                metadatas.append(item["metadata"])
                distances.append(
                    item.get("_additional", {}).get("distance", 0.0))

        return SearchResult(documents=documents,
                            metadatas=metadatas,
                            distances=distances)
