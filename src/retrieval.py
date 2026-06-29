from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document


# get chunks back from chroma
all_docs = vector_store.get()

chunks = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
]

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