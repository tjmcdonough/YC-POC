import streamlit as st
from services.llm_service import LLMService
from services.vector_store import VectorStoreService
from utils.validators import validate_query
from components.results_display import render_results


# TODO: Change this so it rewords the query using an LLM, finds similar vectors, get unique vectors then pass this as context to the langchain LLM
def render_query_interface(vector_store: VectorStoreService,
                           llm_service: LLMService):
    st.header("Query Documents")

    # Single query input field
    query = st.text_area(
        "Enter your query",
        height=100,
        help="Ask any question about your documents",
        placeholder=
        "e.g., 'What are the main topics discussed in the documents?' or 'Find technical specifications for the project'"
    )

    # Search button with loading state
    if st.button("Search", type="primary"):
        if not validate_query(query)[0]:
            st.error("Please enter a valid query")
            return

        with st.spinner("Analyzing query and searching documents..."):
            try:
                # Create similar queries
                queries_string = llm_service.create_similar_queries(query)

                # Get relevant documents with corrected parameter name
                vector_results = vector_store.search(queries_string, top_k=5)

                # Use vector results to pass as context to the LLM
                llm_service.pass_vector_results_as_context(vector_results, queries_string)

                # Display results
                st.success("Search completed!")
                render_results(queries_string, vector_results)
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")
                st.stop()
