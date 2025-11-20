# ingest.py

import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DB_DIR

# Updated FREE local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_documents(file_paths):
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vectordb = Chroma(
        collection_name="docs",
        persist_directory=VECTOR_DB_DIR,     # Auto-persistence
        embedding_function=embedding_model,
    )

    vectordb.add_texts(texts=texts, metadatas=metadatas)

    return vectordb


def load_vector_store():
    vectordb = Chroma(
        collection_name="docs",
        persist_directory=VECTOR_DB_DIR,     # Auto-load
        embedding_function=embedding_model,
    )
    return vectordb
