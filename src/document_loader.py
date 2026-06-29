from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader


def load_documents():
    loader = DirectoryLoader(
        path = r'D:\Astro Books\data\processed' ,
        glob = '*.md' , 
        loader_cls = TextLoader,
        loader_kwargs= {'encoding':'utf-8'}
        
    )
    
    docs = loader.load()
    print(f'Document {len(docs)} Loaded')
    return docs 

if __name__ == "__main__":
    load_documents()


