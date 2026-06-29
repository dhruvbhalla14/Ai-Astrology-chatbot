from vector_store import load_vector
# 1. Call your function to initialize the vector store
vector_store = load_vector()

# 2. Run your search query
query = "Which planet rules discipline?" 

result= vector_store.similarity_search_with_score(
    query= query,
    k = 5
)

print(result)