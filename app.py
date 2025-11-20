# app.py

import os
import json
import streamlit as st

from config import UPLOAD_DIR, VECTOR_DB_DIR
from ingest import load_documents, split_documents, create_vector_store, load_vector_store
from qa_chain import get_qa_chain

# --------------------------------------
# Streamlit Configuration
# --------------------------------------
st.set_page_config(page_title="LangChain Document Q&A", page_icon="📄")
st.title("📄 LangChain Document Q&A Assistant")
st.write("Upload PDF or TXT files and ask questions based on the document content.")


# --------------------------------------
# Session Initialization
# --------------------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    # each item: {"question": str, "answer": str, "sources": [Document, ...]}
    st.session_state.chat_history = []


# --------------------------------------
# Document Upload Section
# --------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload your documents here before asking any question."
)

process_docs = st.button("📥 Process Documents")

if process_docs:
    if uploaded_files:
        st.info("Processing documents... Please wait.")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_paths = []

        # Save uploaded files
        for f in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as file:
                file.write(f.read())
            file_paths.append(save_path)

        # Ingest pipeline
        with st.spinner("🔍 Reading and indexing documents..."):
            docs = load_documents(file_paths)
            chunks = split_documents(docs)
            vectordb = create_vector_store(chunks)
            st.session_state.qa_chain = get_qa_chain(vectordb)

        st.success("Documents processed successfully! You can now ask questions.")
    else:
        st.warning("Please upload at least one document before processing.")


# --------------------------------------
# Load vector store automatically (if exists)
# --------------------------------------
if st.session_state.qa_chain is None and os.path.exists(VECTOR_DB_DIR):
    vectordb = load_vector_store()
    st.session_state.qa_chain = get_qa_chain(vectordb)


# --------------------------------------
# Question Answering Section
# --------------------------------------
st.subheader("💬 Ask Your Question")

user_query = st.text_input("Enter your question:")

ask_button = st.button("Ask")

if ask_button:
    if not st.session_state.qa_chain:
        st.error("Please upload and process documents first.")
    elif not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke(
                user_query,
                chat_history=st.session_state.chat_history
            )

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Save structured item in history
        st.session_state.chat_history.append(
            {
                "question": user_query,
                "answer": answer,
                "sources": sources,
            }
        )


# --------------------------------------
# Chat History + Source Citations
# --------------------------------------
st.subheader("📝 Conversation History")

if len(st.session_state.chat_history) == 0:
    st.info("No conversation yet. Upload documents and ask a question!")
else:
    for idx, item in enumerate(st.session_state.chat_history, start=1):
        q = item["question"]
        a = item["answer"]
        sources = item.get("sources", [])

        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")

        # Source citations
        if sources:
            with st.expander(f"📚 Sources for answer #{idx}"):
                for i, doc in enumerate(sources, start=1):
                    meta = doc.metadata or {}
                    source_path = meta.get("source", "Unknown")
                    page = meta.get("page", "N/A")

                    file_name = os.path.basename(source_path)
                    st.write(f"**Source {i}:** {file_name}, page {page}")

        st.markdown("---")


# --------------------------------------
# Export Conversation History
# --------------------------------------
if st.session_state.chat_history:
    if st.button("💾 Download Conversation (JSON)"):
        # Convert Document objects to serializable form
        export_data = []
        for item in st.session_state.chat_history:
            sources_info = []
            for doc in item.get("sources", []):
                meta = doc.metadata or {}
                sources_info.append(
                    {
                        "source": os.path.basename(meta.get("source", "")),
                        "page": meta.get("page", None),
                    }
                )

            export_data.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "sources": sources_info,
                }
            )

        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Save Conversation as JSON",
            data=json_str,
            file_name="conversation_history.json",
            mime="application/json",
        )


# --------------------------------------
# Clear Button
# --------------------------------------
if st.button("🧹 Clear Conversation"):
    st.session_state.chat_history = []
    st.success("Conversation cleared.")
