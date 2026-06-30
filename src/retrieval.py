from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from vector_store import load_vector
from langchain_core.documents import Document


# Load the persisted Chroma store once and build retrievers from it.
vector_store = load_vector()

# Get chunks back from Chroma.
all_docs = vector_store.get()

chunks = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
]

if not chunks:
    raise ValueError("No indexed documents found in Chroma. Expected data under src/chroma_db.")

# 1. BM25 - keyword based
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 2. Vector - semantic based
vector_retriever = vector_store.as_retriever(
    search_type="mmr",  # mmr reduces redundant chunks
    search_kwargs={
        "k": 5,
        "fetch_k": 20  # fetches 20, then picks best 5 with diversity
    }
)

# 3. Ensemble - combines both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # 60% semantic, 40% keyword
)
