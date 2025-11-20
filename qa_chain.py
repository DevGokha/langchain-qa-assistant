# qa_chain.py

from typing import List, Dict, Any
from langchain_ollama import OllamaLLM


SYSTEM_PROMPT = """
You are a helpful assistant answering questions from uploaded documents.

- Use ONLY the provided document context to answer.
- If the answer is not in the context, reply exactly:
  "The document does not contain this information."
- Use the chat history to understand follow-up questions.
- Be concise and clear.
"""


class QAChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = OllamaLLM(model="llama3")

    def _format_history(self, chat_history: List[Any]) -> str:
        lines = []
        for item in chat_history:
            if isinstance(item, dict):
                q = item.get("question", "")
                a = item.get("answer", "")
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                q, a = item[0], item[1]
            else:
                continue

            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")

        return "\n".join(lines)

    def invoke(self, question: str, chat_history: List[Any] = None) -> Dict[str, Any]:

        # 1. NEW API: retrieve documents
        docs = self.retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        # 2. history formatting
        history_text = ""
        if chat_history:
            history_text = self._format_history(chat_history)

        # 3. Build prompt
        prompt = f"""{SYSTEM_PROMPT}

Chat history:
{history_text}

Document context:
{context}

User question:
{question}

Answer:
"""

        # 4. Get LLM output (string)
        answer = self.llm.invoke(prompt)

        # 5. Always return a dict
        return {
            "answer": answer,
            "sources": docs
        }


def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    return QAChain(retriever)
