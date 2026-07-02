from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from vector_store import vector_store, chunks  # import chunks too for BM25
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from openai import RateLimitError
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# build ensemble retriever here
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

# fixed: {input} not {question}
prompt = ChatPromptTemplate.from_template('''
You are an expert Vedic astrology assistant helping astrologers learn and make predictions.

Use the provided context from classical Vedic astrology texts to answer the question.
The context may use traditional Sanskrit terms or indirect descriptions.
Use your understanding to connect the question with relevant information in the context.

IMPORTANT: Only cite book and chapter information that is explicitly present in the context metadata.
Do NOT invent or guess chapter numbers. If unsure, just mention the book name.

If there is truly nothing relevant in the context, say:
"I couldn't find that information in the provided books."

Context:
{context}

Question:
{input}

Answer:
''')

    



#document_chain = create_stuff_documents_chain(llm, prompt)
#rag_chain = create_retrieval_chain(ensemble_retriever, document_chain)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": ensemble_retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


history = []
MAX_HISTORY = 6  # last 3 exchanges

def chat(query):
    try:
        context_history = "\n".join(history[-MAX_HISTORY:])
        full_input = f"{context_history}\nUser: {query}" if history else query
    
        response = rag_chain.invoke(full_input)
    
        history.append(f"User: {query}")
        history.append(f"Assistant: {response}")
    
        return response
    
    
    except RateLimitError:
        return "❌ OpenAI API quota exceeded. Please try again later."

    except Exception as e:
        return f"❌ Error: {e}"



while True:
    query = input("Prompt: ")
    if query.lower() in ['bye','exit','quit']:
        break
    print(chat(query))
    
    
def chat(query):
    return rag_chain.invoke(query)    
    
'''docs = ensemble_retriever.invoke("Which planet rules discipline?")
for doc in docs:
    print("-" * 60)
    print(doc.metadata)
    print(doc.page_content[:500])
'''

