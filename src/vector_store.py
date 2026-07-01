from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

def create_vector(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="astrology_books",
    )
    return vector_store


def load_vector():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="astrology_books",
    )
    return vector_store


vector_store = load_vector()

all_docs = vector_store.get()
chunks = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
]
