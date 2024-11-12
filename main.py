import streamlit as st
from components.file_upload import render_file_upload
from components.query_interface import render_query_interface
from services.vector_store import VectorStoreService
from services.llm_service import LLMService

st.set_page_config(page_title="Document Processing System",
                   page_icon="📄",
                   layout="wide",
                   initial_sidebar_state="expanded")


def initialize_services():
    try:
        # Initialize services using singleton pattern
        vector_store = VectorStoreService()
        llm_service = LLMService()
        return vector_store, llm_service
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return None, None


def main():
    vector_store, llm_service = initialize_services()
    if not vector_store or not llm_service:
        st.stop()

    st.title("Document Processing System")

    tabs = st.tabs(["Document Upload", "Query Documents"])

    with tabs[0]:
        render_file_upload(vector_store, llm_service)

    with tabs[1]:
        render_query_interface(vector_store, llm_service)


if __name__ == "__main__":
    main()
