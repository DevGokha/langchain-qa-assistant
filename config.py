# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"  # or "gpt-4o-mini"
VECTOR_DB_DIR = "vector_store"
UPLOAD_DIR = "uploaded_docs"
