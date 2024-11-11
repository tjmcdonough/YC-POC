import streamlit as st
from utils.validators import validate_file
import zipfile
import io
from services.file_handler import FileHandlerFactory
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO
import time

def process_chunk(text: str, doc_id: int, chunk_index: int) -> dict:
    """Process a single chunk of text."""
    return {
        "text": text,
        "metadata": {
            "doc_id": doc_id,
            "chunk": chunk_index,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }

def process_single_file(
    file: BinaryIO,
    db_service,
    vector_store,
    llm_service,
    status_container,
    progress_bar,
    batch_size: int = 5
) -> None:
    """Process a single file with consolidated status tracking."""
    try:
        # Get file handler
        file_type = file.name.split('.')[-1].lower()
        handler = FileHandlerFactory.get_handler(file_type)
        
        # Extract and process text
        status_container.info("Extracting text content...")
        text_content = handler.extract_text(file, llm_service)
        
        # Save document to database
        doc_metadata = {
            "filename": file.name,
            "file_type": file_type,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        doc_id = db_service.save_document(
            filename=file.name,
            file_type=file_type,
            summary="",  # Will be updated after processing
            metadata=doc_metadata
        )
        
        # Split text for processing
        chunks = llm_service.split_text(text_content)
        total_chunks = len(chunks)
        
        # Update document with total chunks
        db_service.update_processing_status(
            doc_id=doc_id,
            processed_chunks=0,
            status='processing'
        )
        
        # Process chunks in parallel batches
        processed_chunks = 0
        with ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
            for i in range(0, total_chunks, batch_size):
                current_batch = chunks[i:i + batch_size]
                batch_futures = [
                    executor.submit(process_chunk, chunk, doc_id, i + idx)
                    for idx, chunk in enumerate(current_batch)
                ]
                
                # Process batch results
                batch_results = [f.result() for f in batch_futures]
                vector_store.add_documents(
                    texts=[r["text"] for r in batch_results],
                    metadata=[r["metadata"] for r in batch_results]
                )
                
                # Update progress
                processed_chunks = min(i + batch_size, total_chunks)
                progress = processed_chunks / total_chunks
                progress_bar.progress(progress)
                db_service.update_processing_status(doc_id, processed_chunks)
        
        # Generate and save summary
        summary = llm_service.generate_summary(text_content)
        db_service.update_processing_status(
            doc_id=doc_id,
            processed_chunks=total_chunks,
            status='completed'
        )
        
    except Exception as e:
        status_container.error(f"Error processing file: {str(e)}")
        if 'doc_id' in locals():
            db_service.update_processing_status(
                doc_id=doc_id,
                processed_chunks=0,
                status='failed'
            )
        raise

def render_file_upload(db_service, vector_store, llm_service):
    """Render the file upload interface with simplified progress tracking."""
    st.header("Document Upload")
    
    # File upload interface
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
        # Validate file
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
            return
        
        # Create status tracking
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        try:
            if uploaded_file.name.endswith('.zip'):
                # Process ZIP archive
                with zipfile.ZipFile(uploaded_file) as z:
                    files = [f for f in z.namelist() if not f.endswith('/')]
                    total_files = len(files)
                    
                    for idx, filename in enumerate(files, 1):
                        status_container.info(f"Processing file {idx}/{total_files}: {filename}")
                        with z.open(filename) as f:
                            file_content = io.BytesIO(f.read())
                            file_content.name = filename
                            process_single_file(
                                file_content,
                                db_service,
                                vector_store,
                                llm_service,
                                status_container,
                                progress_bar,
                                batch_size
                            )
            else:
                # Process single file
                process_single_file(
                    uploaded_file,
                    db_service,
                    vector_store,
                    llm_service,
                    status_container,
                    progress_bar,
                    batch_size
                )
            
            status_container.success("File(s) processed successfully!")
        except Exception as e:
            status_container.error(f"Error during processing: {str(e)}")
        finally:
            # Clean up progress indicators after a delay
            time.sleep(2)
            progress_bar.empty()
            status_container.empty()
