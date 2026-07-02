from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path
from vector_store import vector_store, chunks
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from openai import RateLimitError
import os

#load_dotenv(Path(__file__).parent / ".env")  # always finds .env next to rag.py
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

vector_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

prompt = ChatPromptTemplate.from_template("""
You are an expert Vedic astrology assistant.

For greetings or casual conversation, reply naturally without using the context.

For astrology questions:

- Use ONLY the provided context.
- The answer does NOT need to match the user's wording exactly.
- If the context contains information that partially answers the question, use it and explain the connection.
- Infer reasonable conclusions from the classical text, but NEVER introduce knowledge that is not supported by the provided context.
- Never say "I couldn't find that information" if the retrieved context is even partially relevant.
- If multiple passages are relevant, combine them into one coherent answer.
- Quote or summarize the relevant passages in simple language.
- Mention the book and chapter only if they are present in the metadata.

Only if the retrieved context is completely unrelated to the user's question should you answer:

"I couldn't find that information in the provided books."

Conversation History:
{chat_history}

Context:
{context}

Question:
{input}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": (lambda x: x["input"]) | ensemble_retriever | format_docs,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

history = []
MAX_HISTORY = 6

def chat(query):
    try:
        chat_history_text = "\n".join(history[-MAX_HISTORY:]) if history else "No previous conversation."

        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history_text
        })

        history.append(f"User: {query}")
        history.append(f"Assistant: {response}")

        return response

    except RateLimitError:
        return "❌ OpenAI API quota exceeded. Please try again later."
    except Exception as e:
        return f"❌ Error: {e}"


if __name__ == "__main__":
    while True:
        query = input("Prompt: ")
        if query.lower() in ['bye', 'exit', 'quit']:
            break
        print(chat(query))