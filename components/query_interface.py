import streamlit as st
from utils.validators import validate_query
from components.results_display import render_results
from services.file_handler import FileHandlerFactory
from utils.query_templates import QUERY_TEMPLATES
from datetime import datetime, timedelta

def render_query_interface(db_service, vector_store, llm_service):
    st.header("Query Documents")
    
    # Query Type Selection
    query_type = st.radio(
        "Select Query Type",
        ["Free Form", "Template Based"],
        help="Choose between writing your own query or using a template"
    )
    
    if query_type == "Template Based":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Template Selection
            template_key = st.selectbox(
                "Select Query Template",
                options=list(QUERY_TEMPLATES.keys()),
                format_func=lambda x: QUERY_TEMPLATES[x].name,
                help="Choose a predefined query template"
            )
            
            template = QUERY_TEMPLATES[template_key]
            st.info(template.description)
        
        with col2:
            n_results = st.number_input(
                "Number of Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of most relevant results to return"
            )
        
        # Parameter inputs with improved layout
        params = {}
        param_cols = st.columns(min(len(template.parameters), 2))
        
        for idx, param in enumerate(template.parameters):
            col = param_cols[idx % 2]
            with col:
                if param == 'date_range':
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=30)
                    )
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now()
                    )
                    params[param] = f"between {start_date} and {end_date}"
                elif param == 'file_type':
                    params[param] = st.multiselect(
                        "File Types",
                        options=list(FileHandlerFactory._handlers.keys())
                    )
                else:
                    params[param] = st.text_input(
                        f"Enter {param.replace('_', ' ').title()}",
                        help=f"Parameter required for {template.name}"
                    )
        
        # Generate query from template
        if all(params.values()):
            query = template.format_query(params)
            with st.expander("View Generated Query", expanded=False):
                st.code(query, language="text")
        else:
            query = ""
    else:
        # Free Form Query
        query = st.text_area(
            "Enter your query",
            height=100,
            help="Ask questions about your documents",
            placeholder="e.g., What are the main topics discussed in recent documents?"
        )
        n_results = st.number_input(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of most relevant results to return"
        )
    
    # Advanced Filters
    with st.expander("Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            file_type_filter = st.multiselect(
                "Filter by file type",
                options=list(FileHandlerFactory._handlers.keys()),
                help="Select specific file types to search"
            )
        
        with col2:
            date_range = st.date_input(
                "Filter by date range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                help="Select date range for documents"
            )
    
    # Search button with loading state
    if st.button("Search", type="primary", disabled=not query):
        if not validate_query(query)[0]:
            st.error("Please enter a valid query")
            return
            
        with st.spinner("Analyzing query and searching documents..."):
            try:
                # Analyze query intent
                query_analysis = llm_service.analyze_query(query)
                
                # Get relevant documents
                vector_results = vector_store.query_documents(query, n_results=n_results)
                
                # Apply filters
                filters = {}
                if file_type_filter:
                    filters["file_type"] = file_type_filter
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    filters["date_range"] = {
                        "start": date_range[0],
                        "end": date_range[1]
                    }
                
                documents = db_service.get_documents(filters)
                
                # Display results
                st.success("Search completed!")
                render_results(
                    query_analysis,
                    vector_results,
                    documents
                )
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")
                st.stop()
