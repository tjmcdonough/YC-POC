import streamlit as st

def render_results(query_analysis, vector_results):
    st.subheader("Query Analysis")
    st.json(query_analysis)
    
    st.subheader("Relevant Documents")
    
    if not vector_results:
        st.info("No relevant documents found")
        return
        
    for doc in vector_results:
        with st.expander(f"Document: {doc.metadata.get('filename', 'Unknown')}"):
            st.markdown("**Content Extract:**")
            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            
            st.markdown("**Metadata:**")
            st.json(doc.metadata)