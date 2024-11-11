import streamlit as st
from utils.validators import validate_file
import zipfile
import io
from services.file_handler import FileHandlerFactory
import uuid
from queue import Queue
import time
import json
import concurrent.futures
from typing import BinaryIO, List, Dict

def process_chunk(chunk: str, doc_id: int, chunk_index: int) -> Dict:
    """Process a single chunk of text."""
    return {
        "text": chunk,
        "metadata": {
            "doc_id": doc_id,
            "chunk": chunk_index,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        },
        "id": f"{doc_id}-{uuid.uuid4()}"
    }

def render_file_upload(db_service, vector_store, llm_service):
    st.header("Document Upload")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=list(FileHandlerFactory._handlers.keys()) + ['zip'],
            help="Upload documents to process"
        )
    
    with col2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=50,
            value=5,
            help="Number of chunks to process simultaneously"
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
                        llm_service,
                        batch_size
                    )
                else:
                    process_single_file(
                        uploaded_file,
                        db_service,
                        vector_store,
                        llm_service,
                        batch_size
                    )
                st.success("File processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def process_single_file(file: BinaryIO, db_service, vector_store, llm_service, batch_size: int = 5):
    # Create status container
    status_container = st.empty()
    status_container.info("Initializing file processing...")
    
    file_type = file.name.split('.')[-1].lower()
    handler = FileHandlerFactory.get_handler(file_type)
    
    # Extract text content with image analysis if applicable
    status_container.info("Extracting text content...")
    text_content = handler.extract_text(file, llm_service)
    
    # Split text for vector store
    status_container.info("Splitting text into chunks...")
    chunks = llm_service.split_text(text_content)
    total_chunks = len(chunks)
    
    # Save to database with initial status
    doc_id = db_service.save_document(
        filename=file.name,
        file_type=file_type,
        summary="",  # Empty summary since we're not generating one
        metadata=json.dumps({
            "size": len(text_content),
            "chunks": total_chunks,
            "batch_size": batch_size
        }),
        total_chunks=total_chunks
    )
    
    # Create progress tracking
    progress_bar = st.progress(0)
    chunk_status = st.empty()
    
    try:
        # Process chunks in parallel batches
        processed_chunks = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
            # Process chunks in batches
            for i in range(0, total_chunks, batch_size):
                current_batch = chunks[i:i + batch_size]
                chunk_status.text(f"Processing batch {(i//batch_size) + 1} of {(total_chunks + batch_size - 1)//batch_size}")
                
                # Prepare chunks for processing
                futures = [
                    executor.submit(process_chunk, chunk, doc_id, i + idx)
                    for idx, chunk in enumerate(current_batch)
                ]
                
                # Collect processed chunks
                batch_results = []
                for future in concurrent.futures.as_completed(futures):
                    batch_results.append(future.result())
                
                # Add processed chunks to vector store
                vector_store.add_documents(
                    texts=[r["text"] for r in batch_results],
                    metadata=[r["metadata"] for r in batch_results],
                    ids=[r["id"] for r in batch_results]
                )
                
                # Update progress
                processed_chunks = min(i + batch_size, total_chunks)
                progress_bar.progress(processed_chunks / total_chunks)
                db_service.update_processing_status(doc_id, processed_chunks)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
        
        # Mark processing as completed
        db_service.update_processing_status(
            doc_id=doc_id,
            processed_chunks=total_chunks,
            status='completed'
        )
    
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        chunk_status.empty()
        status_container.empty()

def process_zip_file(zip_file: BinaryIO, db_service, vector_store, llm_service, batch_size: int = 5):
    with zipfile.ZipFile(zip_file) as z:
        # Count only files (exclude directories)
        files = [f for f in z.namelist() if not f.endswith('/')]
        total_files = len(files)
        
        if total_files > 0:
            progress_bar = st.progress(0)
            status = st.empty()
            
            for idx, filename in enumerate(files, 1):
                try:
                    status.text(f"Processing file {idx}/{total_files}: {filename}")
                    with z.open(filename) as f:
                        file_content = io.BytesIO(f.read())
                        file_content.name = filename
                        process_single_file(
                            file_content,
                            db_service,
                            vector_store,
                            llm_service,
                            batch_size
                        )
                except Exception as e:
                    st.warning(f"Error processing {filename}: {str(e)}")
                finally:
                    progress_bar.progress(idx / total_files)
            
            progress_bar.empty()
            status.empty()
