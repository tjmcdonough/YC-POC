import streamlit as st
from utils.validators import validate_query
from components.results_display import render_results
from services.file_handler import FileHandlerFactory

def render_query_interface(db_service, vector_store, llm_service):
    st.header("Query Documents")
    
    query = st.text_area(
        "Enter your query",
        height=100,
        help="Ask questions about your documents"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_type_filter = st.multiselect(
            "Filter by file type",
            options=list(FileHandlerFactory._handlers.keys())
        )
    
    with col2:
        date_range = st.date_input(
            "Filter by date range",
            value=[None, None]
        )
    
    if st.button("Search", type="primary"):
        if not validate_query(query)[0]:
            st.error("Please enter a valid query")
            return
            
        with st.spinner("Searching..."):
            # Analyze query intent
            query_analysis = llm_service.analyze_query(query)
            
            # Get relevant documents
            vector_results = vector_store.query_documents(query)
            
            # Get document metadata
            filters = {}
            if file_type_filter:
                filters["file_type"] = file_type_filter
            if date_range[0] and date_range[1]:
                filters["date_range"] = {
                    "start": date_range[0],
                    "end": date_range[1]
                }
            
            documents = db_service.get_documents(filters)
            
            render_results(
                query_analysis,
                vector_results,
                documents
            )
