import streamlit as st
from utils.validators import validate_file
import zipfile
import io
from services.file_handler import FileHandlerFactory
import uuid
import threading
from queue import Queue
import time
import json

def render_file_upload(db_service, vector_store, llm_service):
    st.header("Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=list(FileHandlerFactory._handlers.keys()) + ['zip'],
        help="Upload documents to process"
    )

    if uploaded_file:
        is_valid, error_msg = validate_file(uploaded_file)
        
        if not is_valid:
            st.error(error_msg)
            return

        with st.spinner("Processing file..."):
            try:
                if uploaded_file.name.endswith('.zip'):
                    process_zip_file(
                        uploaded_file,
                        db_service,
                        vector_store,
                        llm_service
                    )
                else:
                    process_single_file(
                        uploaded_file,
                        db_service,
                        vector_store,
                        llm_service
                    )
                st.success("File processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def process_single_file(file, db_service, vector_store, llm_service):
    file_type = file.name.split('.')[-1].lower()
    handler = FileHandlerFactory.get_handler(file_type)
    
    # Extract text content with image analysis if applicable
    text_content = handler.extract_text(file, llm_service)
    
    # Generate initial summary
    summary = "Processing..." if len(text_content) > 10000 else llm_service.generate_summary(text_content)
    
    # Split text for vector store
    chunks = llm_service.split_text(text_content)
    total_chunks = len(chunks)
    
    # Save to database with initial status
    metadata = json.dumps({"size": len(text_content)})
    doc_id = db_service.save_document(
        filename=file.name,
        file_type=file_type,
        summary=summary,
        metadata=metadata,
        total_chunks=total_chunks
    )
    
    # Create progress bar if needed
    progress_bar = None
    if total_chunks > 1:
        progress_bar = st.progress(0)
        st.text("Processing chunks...")
    
    # Process chunks in batches
    batch_size = 5
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"{doc_id}-{uuid.uuid4()}" for _ in range(len(batch_chunks))]
        batch_metadata = [{"doc_id": doc_id, "chunk": i+j} for j in range(len(batch_chunks))]
        
        # Save batch to vector store
        vector_store.add_documents(
            texts=batch_chunks,
            metadata=batch_metadata,
            ids=batch_ids
        )
        
        # Update progress
        processed_chunks = min(i + batch_size, total_chunks)
        if progress_bar:
            progress_bar.progress(processed_chunks / total_chunks)
        db_service.update_processing_status(doc_id, processed_chunks)
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    # Update final summary if it was initially deferred
    if len(text_content) > 10000:
        summary = llm_service.generate_summary(text_content)
        db_service.update_processing_status(doc_id, total_chunks, 'completed')
        
    if progress_bar:
        progress_bar.empty()

def process_zip_file(zip_file, db_service, vector_store, llm_service):
    with zipfile.ZipFile(zip_file) as z:
        total_files = len([f for f in z.namelist() if not f.endswith('/')])
        if total_files > 0:
            progress_bar = st.progress(0)
            st.text(f"Processing {total_files} files...")
            
            for idx, filename in enumerate(z.namelist()):
                if not filename.endswith('/'):  # Skip directories
                    with z.open(filename) as f:
                        file_content = io.BytesIO(f.read())
                        file_content.name = filename
                        process_single_file(
                            file_content,
                            db_service,
                            vector_store,
                            llm_service
                        )
                        progress_bar.progress((idx + 1) / total_files)
            
            progress_bar.empty()
