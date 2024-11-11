import streamlit as st
from utils.validators import validate_file
import zipfile
import io
from services.file_handler import FileHandlerFactory
from typing import BinaryIO

def process_single_file(file: BinaryIO, vector_store, llm_service) -> None:
    """Process a single file by extracting text and adding to vector store."""
    # Get file handler based on file type
    file_type = file.name.split('.')[-1].lower()
    handler = FileHandlerFactory.get_handler(file_type)
    
    # Extract text content from the file
    text_content = handler.extract_text(file, llm_service)
    
    # Add extracted text to vector store
    vector_store.add_documents(
        texts=[text_content],
        metadata=[{"filename": file.name, "file_type": file_type}]
    )

def render_file_upload(vector_store, llm_service):
    """Render file upload interface for Streamlit application."""
    # Display header for file upload section
    st.header("Document Upload")
    
    # Create file uploader with supported file types
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=list(FileHandlerFactory._handlers.keys()) + ['zip'],
        help="Upload documents to process"
    )
    
    # Process uploaded file if present
    if uploaded_file:
        # Validate uploaded file
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
            return
        
        try:
            # Handle ZIP file processing
            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file) as z:
                    # Process each file in the ZIP archive
                    for filename in z.namelist():
                        if filename.endswith('/'):  # Skip directories
                            continue
                        with z.open(filename) as f:
                            file_content = io.BytesIO(f.read())
                            file_content.name = filename
                            process_single_file(file_content, vector_store, llm_service)
            else:
                # Process single file
                process_single_file(uploaded_file, vector_store, llm_service)
            
            # Display success message
            st.success("File(s) processed successfully!")
        
        # Handle any processing errors
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")