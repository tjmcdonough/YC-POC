import json
from typing import List, Dict, Any


class VectorDocument:

    def __init__(self, id: str, vector: List[float], metadata: Dict[str, Any]):
        self.id = id
        self.vector = vector
        self.metadata = metadata

    @classmethod
    def from_json(cls, json_str: str) -> 'VectorDocument':
        """Create a VectorDocument from a JSON string"""
        data = json.loads(json_str)
        return cls(id=data['id'],
                   vector=data['vector'],
                   metadata=data['metadata'])

    def to_json(self) -> str:
        """Convert the document to a JSON string"""
        return json.dumps({
            'id': self.id,
            'vector': self.vector,
            'metadata': self.metadata
        })
