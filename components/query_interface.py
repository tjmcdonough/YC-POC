import streamlit as st
from utils.validators import validate_query
from components.results_display import render_results
from services.file_handler import FileHandlerFactory
from datetime import datetime, timedelta


def render_query_interface(vector_store, llm_service):
    st.header("Query Documents")

    # Single query input field
    query = st.text_area(
        "Enter your query",
        height=100,
        help="Ask any question about your documents",
        placeholder=
        "e.g., 'What are the main topics discussed in the documents?' or 'Find technical specifications for the project'"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        # Advanced Filters in expandable section
        with st.expander("Advanced Filters"):
            file_type_filter = st.multiselect(
                "Filter by file type",
                options=list(FileHandlerFactory._handlers.keys()),
                help="Select specific file types to search")

            date_range = st.date_input(
                "Filter by date range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                help="Select date range for documents")

    with col2:
        n_results = st.number_input(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of most relevant results to return")

    # Search button with loading state
    if st.button("Search", type="primary"):
        if not validate_query(query)[0]:
            st.error("Please enter a valid query")
            return

        with st.spinner("Analyzing query and searching documents..."):
            try:
                # Analyze query intent
                query_analysis = llm_service.analyze_query(query)

                # Get relevant documents
                vector_results = vector_store.search(query, top_k=n_results)

                # Filter results based on user selection
                filtered_results = []
                for doc in vector_results:
                    if file_type_filter and doc.metadata.get('file_type') not in file_type_filter:
                        continue
                    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
                        doc_date = datetime.fromtimestamp(float(doc.metadata.get('created_at', 0)))
                        if not (date_range[0] <= doc_date.date() <= date_range[1]):
                            continue
                    filtered_results.append(doc)

                # Display results
                st.success("Search completed!")
                render_results(query_analysis, filtered_results)
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")
                st.stop()
