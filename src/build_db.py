from document_loader import load_documents
from Text_Splitter import split_documents
from vector_store import create_vector

docs = load_documents()
chunks = split_documents(docs)
vector_store = create_vector(chunks)

print("Vector DB Created Successfully!")