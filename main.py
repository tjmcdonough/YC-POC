import streamlit as st
from components.file_upload import render_file_upload
from components.query_interface import render_query_interface
from services.database import DatabaseService
from services.vector_store import VectorStoreService
from services.llm_service import LLMService

st.set_page_config(
    page_title="Document Processing System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_services():
    try:
        if 'db_service' not in st.session_state:
            st.session_state.db_service = DatabaseService()
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = VectorStoreService()
        if 'llm_service' not in st.session_state:
            st.session_state.llm_service = LLMService()
        return True
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return False

def main():
    if not initialize_services():
        st.stop()
    
    st.title("Document Processing System")
    
    tabs = st.tabs(["Document Upload", "Query Documents"])
    
    with tabs[0]:
        render_file_upload(
            st.session_state.db_service,
            st.session_state.vector_store,
            st.session_state.llm_service
        )
    
    with tabs[1]:
        render_query_interface(
            st.session_state.db_service,
            st.session_state.vector_store,
            st.session_state.llm_service
        )

if __name__ == "__main__":
    main()
