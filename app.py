import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# --- PAGE CONFIGURATION ---
# Set layout to "centered" which looks great on both desktop and mobile
st.set_page_config(page_title="Resume Chatbot", page_icon="📄", layout="centered")

# --- SESSION STATE INITIALIZATION ---
# This prevents the app from forgetting data when it re-renders
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# --- SIDEBAR (Settings & File Upload) ---
# Moving inputs to the sidebar frees up screen real estate on mobile devices
with st.sidebar:
    st.header("⚙️ Setup & Configuration")
    
    groq_api_key = st.text_input("Enter Groq API Key", type="password", help="Get this from console.groq.com")
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"],
        help="Upload a PDF resume to start chatting."
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("📄 Resume RAG Chatbot")
st.markdown("Ask anything about the uploaded resume! (e.g., *'What are their main skills?'* or *'Summarize their work experience.'*)")

# --- DOCUMENT PROCESSING ---
if uploaded_file and groq_api_key:
    # Only process the file if it's newly uploaded (saves time and resources)
    if st.session_state.processed_file != uploaded_file.name:
        
        # UI Note while processing
        with st.spinner("⏳ Analyzing resume and building memory... This might take a few seconds."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    pdf_path = tmp_file.name

                # Load and split document
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                docs = splitter.split_documents(documents)

                # Generate Embeddings & Vector Store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)

                # Initialize LLM
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama-3.3-70b-versatile"
                )

                # Create RAG Chain (Using invoke instead of deprecated run)
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                )

                # Update session state so we don't re-process this same file
                st.session_state.processed_file = uploaded_file.name
                st.session_state.chat_history = [] # Reset chat for the new file
                
                # Clean up temporary file
                os.remove(pdf_path)
                
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.session_state.qa_chain = None
                
        st.success(f"✅ Successfully processed **{uploaded_file.name}**! You can now chat.")

elif not groq_api_key:
    st.info("👈 Please enter your **Groq API Key** in the sidebar to get started.")
elif not uploaded_file:
    st.info("👈 Please upload a **PDF Resume** in the sidebar.")

# --- CHAT DISPLAY & INTERACTION ---
st.divider()

# 1. Display previous chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Chat Input field (locked at the bottom, perfectly responsive for mobile)
if st.session_state.qa_chain:
    prompt = st.chat_input("Ask a question about the resume...")
    
    if prompt:
        # Display user message instantly
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display Assistant response with a thinking spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                try:
                    # Invoke the chain
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = response.get("result", "I couldn't find an answer to that in the resume.")
                    
                    st.markdown(answer)
                    
                    # Save AI message to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {e}")