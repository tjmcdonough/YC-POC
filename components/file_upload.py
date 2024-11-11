import streamlit as st
from utils.validators import validate_file
import zipfile
import io
from services.file_handler import FileHandlerFactory
import uuid

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
    
    # Extract text content
    text_content = handler.extract_text(file)
    
    # Generate summary using LLM
    summary = llm_service.generate_summary(text_content)
    
    # Split text for vector store
    chunks = llm_service.split_text(text_content)
    
    # Save to database
    doc_id = db_service.save_document(
        filename=file.name,
        file_type=file_type,
        summary=summary,
        metadata={"size": len(text_content)}
    )
    
    # Save to vector store
    vector_store.add_documents(
        texts=chunks,
        metadata=[{"doc_id": doc_id, "chunk": i} for i in range(len(chunks))],
        ids=[f"{doc_id}-{uuid.uuid4()}" for _ in range(len(chunks))]
    )

def process_zip_file(zip_file, db_service, vector_store, llm_service):
    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
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
