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
                if full_doc['processing_status'] == 'processing':
                    st.warning(f"Processing: {full_doc['processed_chunks']}/{full_doc['total_chunks']} chunks")
                else:
                    st.write(full_doc['summary'])
    
    st.subheader("All Documents")
    df = pd.DataFrame(documents)
    df['status'] = df.apply(
        lambda x: f"{x['processing_status']} ({x['processed_chunks']}/{x['total_chunks']})" 
        if x['processing_status'] == 'processing' 
        else 'completed',
        axis=1
    )
    st.dataframe(
        df[['filename', 'file_type', 'status', 'created_at']],
        use_container_width=True
    )
