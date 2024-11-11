from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

@dataclass
class Document:
    filename: str
    file_type: str
    content: str
    summary: Optional[str] = None
    metadata: Dict = None
    created_at: datetime = None
    id: Optional[int] = None
