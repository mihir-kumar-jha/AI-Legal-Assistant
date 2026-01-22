from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables from .env file...")
load_dotenv()

#Step1: Setup LLM (Use DeepSeek R1 with Groq)
logger.info("Initializing ChatGroq LLM model...")
llm_model=ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
logger.info("✓ ChatGroq LLM model initialized successfully")

#Step2: Retrieve Docs

def retrieve_docs(query):
    logger.info(f"Retrieving documents for query: '{query}'")
    docs = faiss_db.similarity_search(query)
    logger.info(f"✓ Retrieved {len(docs)} relevant documents")
    return docs

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#Step3: Answer Question

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    logger.info("Building prompt and invoking LLM...")
    context = get_context(documents)
    logger.info(f"Context length: {len(context)} characters from {len(documents)} documents")
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    logger.info("Invoking ChatGroq model for answer generation...")
    response = chain.invoke({"question": query, "context": context})
    logger.info("✓ Answer generated successfully")
    return response

logger.info("=" * 80)
logger.info("Starting RAG Pipeline Execution")
logger.info("=" * 80)

# question="Explain the article 15."
# logger.info(f"User Question: {question}")
# logger.info("-" * 80)

# retrieved_docs=retrieve_docs(question)
# logger.info("-" * 80)

# response = answer_query(documents=retrieved_docs, model=llm_model, query=question)
# logger.info("=" * 80)
# logger.info("RESPONSE FROM AI LAWYER:")
# logger.info("=" * 80)
# print(response)
# logger.info("=" * 80)
# logger.info("Pipeline Execution Complete")
# logger.info("=" * 80)