import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_docs(path="docs"): 
    print(f"loading the files")
    if not os.path.exists(path): 
        raise FileNotFoundError(f"the directory not exist {path}")

    loader=DirectoryLoader(path=path,
                           glob=["*.txt","*.pdf"],
                           loader_cls=TextLoader)
    documents=loader.load()

    if len(documents)==0: 
        raise FileNotFoundError(f"No .txt or .pdf files found at {path}")
    
    for i, doc in enumerate(documents[:2]): 
        print(f"Document is {i+1}\n")
        print(f"source : {doc.metadata['source']}")
        print(f"content length: {len(doc.page_content)} characters")
        print(f" content preview: {doc.page_content[:100]}")
        print(f" meta data {doc.metadata}")
    
    return documents

def split_documents(documents,chunk_size=1000,chunk_overlap=0): 
    print(f"chunking fiels....")

    text_splitters=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks=text_splitters.split_documents(documents)

    if chunks: 
        for i ,chunk in enumerate(chunks[:5]):
            print(f"-- chunk {i+1}")
            print(f"source: {chunk.metadata['source']}")
            print(f"length:{len(chunk.page_content)} characters")
            print(f"contnte: ")
            print(chunk.page_content)
            print("-----"*20)

        if(len(chunks)>5): 
            print(f"\n and {len(chunks)-5} more ...")

    return chunks  


def create_vector_store(chunks,presist_directory="db/chromadb"):
    print(f"creating embeddings and storing in chromaDB ...")

    embedding_model=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    print("--creting vector db---")
    
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=presist_directory,
        collection_metadata={"hnsw:space":"cosine"}
        
    )
    print("--vector db created")
    print(f" db cretead at {presist_directory}")
    return vectorstore


def main(): 
    print("main function ")
    documents=load_docs("/Users/vikram/Documents/RAG/data/text_files")
    chunks=split_documents(documents)
    vectorstore=create_vector_store(chunks)


if __name__=="__main__": 
    main()