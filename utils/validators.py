from typing import BinaryIO
from utils.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from urllib.parse import urlparse
import re

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

def validate_url(url: str) -> tuple[bool, str]:
    """Validate URL format and structure"""
    if not url:
        return False, "URL cannot be empty"
        
    # Check URL length
    if len(url) > 2048:
        return False, "URL is too long"
        
    # Check URL format
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False, "Invalid URL format"
            
        # Check if scheme is http or https
        if result.scheme not in ['http', 'https']:
            return False, "URL must start with http:// or https://"
            
        # Basic domain validation
        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        )
        if not domain_pattern.match(result.netloc):
            return False, "Invalid domain name"
            
        return True, ""
        
    except Exception:
        return False, "Invalid URL format"
