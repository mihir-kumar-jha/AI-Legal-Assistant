from rag_pipeline import answer_query, retrieve_docs, llm_model
import streamlit as st
import logging

# Configure logging for Streamlit
logger = logging.getLogger(__name__)
logger.info("Frontend initialized")

uploaded_file = st.file_uploader("Upload PDF",
                                 type="pdf",
                                 accept_multiple_files=False)


#Step2: Chatbot Skeleton (Question & Answer)

user_query = st.text_area("Enter your prompt: ", height=150 , placeholder= "Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:

    if uploaded_file: 
        logger.info(f"Processing query: {user_query[:50]}...")
        st.chat_message("user").write(user_query)

        # RAG Pipeline
        logger.info("Starting RAG retrieval...")
        retrieved_docs=retrieve_docs(user_query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        logger.info("Generating response from AI Lawyer...")
        response=answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        logger.info("Response generated successfully")
        
        st.chat_message("AI Lawyer").write(response)
    
    else:
        logger.warning("No PDF file uploaded")
        st.error("Kindly upload a valid PDF file first!")