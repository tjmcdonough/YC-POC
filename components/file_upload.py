import streamlit as st
import zipfile
import io
from typing import BinaryIO, List
from utils.validators import validate_file, validate_url
from services.file_handler import FileHandlerFactory
from services.web_scraper import WebScraperService, crawl_website
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
    vector_store.add_documents(text=text_content,
                               metadata={
                                   "filename": file.name,
                                   "file_type": file_type,
                                   "created_at": timestamp
                               })


def process_url(url: str, vector_store, llm_service) -> None:
    web_scraper = WebScraperService()
    scraped_results = web_scraper.crawl_website(url, vector_store, llm_service)

    for result in scraped_results:
        print(result)


def render_file_upload(vector_store, llm_service):
    st.header("Add Documents")

    # Add Clear Data button
    if st.button("Clear All Documents", type="secondary"):
        try:
            vector_store.clear_data()
            st.success("All documents cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")

    # Create tabs for different input methods
    upload_tab, url_tab = st.tabs(["File Upload", "URL Input"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=list(FileHandlerFactory._handlers.keys()) + ['zip'],
            help="Upload documents to process")

        if uploaded_file:
            # Validate file
            is_valid, error_msg = validate_file(uploaded_file)
            if not is_valid:
                st.error(error_msg)
                return

            try:
                if uploaded_file.name.endswith('.zip'):
                    with zipfile.ZipFile(uploaded_file) as z:
                        for filename in z.namelist():
                            if filename.endswith('/'):  # Skip directories
                                continue
                            with z.open(filename) as f:
                                file_content = io.BytesIO(f.read())
                                file_content.name = filename
                                process_single_file(file_content, vector_store,
                                                    llm_service)
                else:
                    process_single_file(uploaded_file, vector_store,
                                        llm_service)

                st.success("File(s) processed successfully!")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

    with url_tab:
        url_input = st.text_input(
            "Enter root URL",
            help="Enter web page root URL to scrape and process",
            placeholder="https://example.com")

        if st.button("Process URLs", type="primary"):

            if not url_input:
                st.error("Please enter a URL")
                return

            try:
                with st.spinner("Processing URLs..."):
                    process_url(url_input, vector_store, llm_service)
                st.success(f"Successfully processed {url_input} URLs!")
            except Exception as e:
                st.error(f"Error processing URLs: {str(e)}")
