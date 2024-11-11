from typing import BinaryIO
from utils.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE

def validate_file(file: BinaryIO) -> tuple[bool, str]:
    # Check file extension
    if not file.name.lower().split('.')[-1] in ALLOWED_EXTENSIONS:
        return False, "Invalid file type"

    # Check file size
    file.seek(0, 2)  # Seek to end of file
    size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if size > MAX_FILE_SIZE:
        return False, "File too large"

    return True, ""

def validate_query(query: str) -> tuple[bool, str]:
    if not query.strip():
        return False, "Query cannot be empty"
    
    if len(query) > 1000:
        return False, "Query too long"
        
    return True, ""
