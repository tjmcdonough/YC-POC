import streamlit as st
import zipfile
import io
from typing import BinaryIO
from utils.validators import validate_file
from services.file_handler import FileHandlerFactory
import time

def process_single_file(file: BinaryIO, vector_store, llm_service) -> None:
    # Get file handler
    file_type = file.name.split('.')[-1].lower()
    handler = FileHandlerFactory.get_handler(file_type)
    
    # Extract text content
    text_content = handler.extract_text(file, llm_service)
    
    # Get timestamp now
    timestamp = time.time()
    
    # Add to vector store
    vector_store.add_documents(
        text=text_content,
        metadata={"filename": file.name, "file_type": file_type, "created_at": timestamp}
    )

def render_file_upload(vector_store, llm_service):
    st.header("Document Upload")
    
    # Add Clear Data button
    if st.button("Clear All Documents", type="secondary"):
        try:
            vector_store.clear_data()
            st.success("All documents cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=list(FileHandlerFactory._handlers.keys()) + ['zip'],
        help="Upload documents to process"
    )
    
    if uploaded_file:
        # Validate file
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
            return
            
        print("Uploading file...")

        try:
            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file) as z:
                    for filename in z.namelist():
                        if filename.endswith('/'):  # Skip directories
                            continue
                        with z.open(filename) as f:
                            file_content = io.BytesIO(f.read())
                            file_content.name = filename
                            process_single_file(file_content, vector_store, llm_service)
            else:
                process_single_file(uploaded_file, vector_store, llm_service)
            
            st.success("File(s) processed successfully!")
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
