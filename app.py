import streamlit as st
import os
from data_loader import process_all_pdfs, split_documents
from embeddings import EmbeddingManager
from vectorstore import VectorStore
from rag_retriever import RAGRetriever
from rag import rag_advance

st.set_page_config(page_title="RAG PDF QA Assistant", layout="wide")
st.title("📄 RAG PDF QA Assistant")

DATA_DIR = "data/uploaded_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Sidebar - File Manager
st.sidebar.header("📚 Document Library")
pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]

if pdf_files:
    for f in pdf_files:
        st.sidebar.write(f"• {f}")
else:
    st.sidebar.write("No files uploaded yet.")

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join(DATA_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
    st.sidebar.success("Files uploaded and saved!")

# Build Knowledge Base
if st.sidebar.button("🔄 Build / Update Knowledge Base"):
    with st.spinner("Processing PDFs and generating embeddings..."):
        all_docs = process_all_pdfs(DATA_DIR)
        chunks = split_documents(all_docs)
        texts = [doc.page_content for doc in chunks]

        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.generate_embeddings(texts)

        vectorstore = VectorStore()
        vectorstore.add_documents(chunks, embeddings)

        st.session_state.retriever = RAGRetriever(vectorstore, embedding_manager)

    st.sidebar.success("Knowledge Base Updated!")

# init chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat
for idx, (sender, message) in enumerate(st.session_state.chat_history):
    if sender == "You":
        st.chat_message("user").markdown(message)

    else:
        st_chat = st.chat_message("assistant")
        st_chat.markdown(message["answer"])

        # Sources button
        if "sources" in message:
            if st.button("📌 Sources", key=f"sources_btn_{idx}"):
                message["show_sources"] = not message["show_sources"]
                st.session_state.chat_history[idx] = (sender, message)
                st.rerun()

            if message.get("show_sources", False):
                for i, src in enumerate(message["sources"]):
                    with st.expander(f"Source {i+1} — Page {src['page']+1} — Score {src['score']:.3f}"):
                        st.write("**File:**", src["source"])
                        st.write(src["preview"])

# input
query = st.chat_input("Ask a question about your PDFs...")

if query and "retriever" in st.session_state:
    result = rag_advance(
        query,
        st.session_state.retriever,
        top_k=3,
        return_context=True
    )

    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Assistant", {
        "answer": result["answer"],
        "sources": result["sources"],
        "show_sources": False
    }))
    st.rerun()
