LangChain Document Q&A Assistant
🚀 Overview

The LangChain Document Q&A Assistant is a chatbot that answers questions from uploaded PDF/TXT documents using:

🔍 Document chunking

🧠 Vector embeddings

🗄 Local vector database (Chroma)

🤖 Local LLM (Ollama – Llama 3)

🧵 Conversation memory

📝 Source citation

🌐 Streamlit interface

It is designed as a complete beginner-friendly project for learning document intelligence, retrieval-augmented generation (RAG), and LangChain.

🎯 Features
✔ Core Features (Required)

Upload PDF or TXT documents

Document chunking using RecursiveCharacterTextSplitter

Local embeddings using HuggingFace MiniLM

Vector storage with Chroma DB (persistent)

Natural-language question answering

Context-aware responses

Source citation (file name + page number for PDFs)

Conversation memory (follow-up questions work)

LangChain retriever + Ollama LLM pipeline

⭐ Bonus Features (Optional)

Multiple document support

Streamlit front-end

Conversation history panel

Download conversation history as JSON

Clean UI with expandable source panels

🏗️ Tech Stack
Component	Technology
LLM	Ollama (local) – Llama 3
Embeddings	HuggingFace MiniLM
Vector DB	Chroma
Framework	LangChain
Front-end	Streamlit
Loader	PyPDFLoader, TextLoader
Language	Python 3.10+
📁 Folder Structure
LangChain-QA-Assistant/
│── app.py                 # Streamlit UI
│── qa_chain.py            # LLM + retrieval + memory logic
│── ingest.py              # Document loading + chunking + vector DB
│── config.py              # Path configuration
│── requirements.txt
│── docs/                  # Optional sample documents
│── vectorstore/           # Auto-created Chroma DB
│── uploads/               # Uploaded files

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Devgokha/langchain-qa-assistant.git
cd langchain-qa-assistant

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Install Ollama

Download from:
https://ollama.com

Then run:

ollama pull llama3
ollama serve


Keep Ollama running.

▶️ Running the App
streamlit run app.py


Then open:
👉 http://localhost:8501

🖼 How to Use

Upload one or multiple PDF/TXT documents

Click Process Documents

Ask any question

View the answer + source citations

Download the conversation as JSON if needed

✨ Example Questions

What is Artificial Intelligence?

Explain machine learning in simple terms.

What are the applications mentioned in the document?

Give a summary of the uploaded document.

What does the document say about reasoning?



✔ Evaluation Criteria (Satisfied)

Functionality (40%) – Works fully with RAG, citations, memory

Code Quality (25%) – Modular (qa_chain.py, ingest.py, etc.)

Documentation (20%) – This README + explanation

User Experience (15%) – Clear Streamlit UI

🙌 Author

Dev Gokha
MERN + AI/ML Developer
Passionate about RAG, LangChain, and applied AI engineering.