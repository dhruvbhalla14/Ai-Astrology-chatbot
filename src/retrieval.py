# use this combination
# 1. semantic search via chromadb
# 2. BM25 keyword search
# 3. combine with RRF (Reciprocal Rank Fusion)

from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # slightly favor semantic
)