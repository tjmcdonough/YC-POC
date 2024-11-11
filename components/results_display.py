import streamlit as st
import pandas as pd

def render_results(query_analysis, vector_results, documents):
    st.subheader("Query Analysis")
    st.json(query_analysis)
    
    st.subheader("Relevant Documents")
    
    if not vector_results['documents']:
        st.info("No relevant documents found")
        return
        
    for doc, metadata, score in zip(
        vector_results['documents'],
        vector_results['metadatas'],
        vector_results['distances']
    ):
        with st.expander(f"Document {metadata['doc_id']} (Score: {1 - score:.2f})"):
            st.markdown("**Content Extract:**")
            st.text(doc[:500] + "..." if len(doc) > 500 else doc)
            
            st.markdown("**Metadata:**")
            st.json(metadata)
            
            # Find full document info
            full_doc = next(
                (d for d in documents if d['id'] == metadata['doc_id']),
                None
            )
            if full_doc:
                st.markdown("**Document Summary:**")
                st.write(full_doc['summary'])
    
    st.subheader("All Documents")
    df = pd.DataFrame(documents)
    st.dataframe(
        df[['filename', 'file_type', 'created_at']],
        use_container_width=True
    )
