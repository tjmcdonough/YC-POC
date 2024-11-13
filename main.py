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

    st.title("AI Document Processing & Analysis System")
    st.text(
        "Hey YC! This is a very basic generic RAG system. The customer flow will work like this. User types in what product they are building, we generate agents that handle specific tasks via an in depth prompt, they add their data via URL if they want us to scrape or attach, we tweak the infrastructure if we need to. Then the user just plugs in an API and can ask anything using text and images"
    )

    tabs = st.tabs(["Document Upload", "Query Documents"])

    with tabs[0]:
        render_file_upload(vector_store, llm_service)

    with tabs[1]:
        render_query_interface(vector_store, llm_service)


if __name__ == "__main__":
    main()
