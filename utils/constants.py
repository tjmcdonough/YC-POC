ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'jpg', 'jpeg', 'png',
    'json', 'xml', 'csv', 'zip'
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

ERROR_MESSAGES = {
    'file_type': 'Unsupported file type. Please upload a supported file.',
    'file_size': 'File too large. Maximum size is 10MB.',
    'processing': 'Error processing file. Please try again.',
}
