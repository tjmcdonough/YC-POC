import streamlit as st

def render_results(query_analysis, vector_results, documents):
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

    st.subheader("All Documents")
    df = pd.DataFrame(
        [
            {
                "filename": doc.metadata.get('filename', 'Unknown'),
                "file_type": doc.metadata.get('file_type', 'Unknown'),
                "status": "completed",
                "created_at": doc.metadata.get('created_at', 'Unknown')
            }
            for doc in documents
        ]
    )
    st.dataframe(
        df[['filename', 'file_type', 'status', 'created_at']],
        use_container_width=True
    )