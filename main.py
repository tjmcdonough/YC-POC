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

    st.subheader("Hey YC")
    st.write("This is a very basic generic RAG system. An actual MVP will be ready to showcase the below by end of November. It will work something like...")

    st.subheader("Customer Flow")

    st.markdown("""
    1. **Define the Product**: The user describes their product goals.
    2. **Auto-Generate Agents**: We craft tailored agents for specific tasks based on a comprehensive prompt.
    3. **Data Integration**: Users can provide data via URLs (for web scraping) or attachments, simplifying setup.
    4. **Infrastructure Tuning**: We handle any necessary infrastructure adjustments on our end. The developer can also make tweaks their end for maximum customisation.
    5. **Single API Access**: The user connects to our main API, gaining the ability to query their data via text and images seamlessly.
    """)

    st.write("This approach brings AI integration into reach without the typical complexity.")

    tabs = st.tabs(["Document Upload", "Query Documents"])

    with tabs[0]:
        render_file_upload(vector_store, llm_service)

    with tabs[1]:
        render_query_interface(vector_store, llm_service)


if __name__ == "__main__":
    main()
