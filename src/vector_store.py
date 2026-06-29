from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vector(chunks):
    embeddings =  OpenAIEmbeddings(model ='text-embedding-3-small')
    
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory= 'chroma_db',
        collection_name= 'astrology_books'
    )
    return vector_store


def load_vector():
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
    vector_store = Chroma(
        persist_directory= 'chroma_db',
        embedding_function = embeddings,
        collection_name= 'astrology_books'
    )    
    return vector_store
