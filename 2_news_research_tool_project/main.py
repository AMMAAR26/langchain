import os
import streamlit as st
import pickle
import time
import google.generativeai as palm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env (especially GOOGLE_API_KEY)

# Configure Google PaLM API
palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_google.pkl"

main_placeholder = st.empty()

# Use SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_palm_response(prompt):
    """Generates a response using Google PaLM 2."""
    response = palm.generate_text(model="models/text-bison-001", prompt=prompt)
    return response.result if response.result else "No response."

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save them to FAISS index
    doc_texts = [doc.page_content for doc in docs]
    embeddings = embedder.encode(doc_texts, show_progress_bar=True)

    vectorstore_google = FAISS.from_embeddings(embeddings, doc_texts)
