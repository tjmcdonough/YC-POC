ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'jpg', 'jpeg', 'png',
    'json', 'xml', 'csv', 'zip', 'md',
    'txt', 'html', 'htm', 'rtf'
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

ERROR_MESSAGES = {
    'file_type': 'Unsupported file type. Please upload a supported file.',
    'file_size': 'File too large. Maximum size is 10MB.',
    'processing': 'Error processing file. Please try again.',
    'url_invalid': 'Invalid URL format.',
    'url_unreachable': 'Unable to access the URL.',
    'url_timeout': 'Request timed out.',
}

# Web scraping constants
MAX_URLS_PER_BATCH = 10
REQUEST_TIMEOUT = 10  # seconds
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
