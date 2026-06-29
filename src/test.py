from vector_store import load_vector

# 1. Call your function to initialize the vector store
vector_store = load_vector()


# test it
query = "Which planet rules discipline?"
results = ensemble_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n[{i}] {doc.metadata['book']} - {doc.metadata['chapter']}")
    print(doc.content[:200])