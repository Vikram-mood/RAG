from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage 

load_dotenv()

persist_directory="db/chromadb"

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db=Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space':'cosine'}

)
query="tyeps of machine learning"

retriever=db.as_retriever(search_kwargs={'k':3})

# retriver=db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         'k':5,
#         'score_threshold':0.3
#     }
# )


relevent_doc=retriever.invoke(query)




print(f"use query : {query}")
print(f"-------context-----")

for i, doc in enumerate(relevent_doc,1): 
    print(f"Document  {i}: \n {doc.page_content}\n")


combined_input= f"""Based on the following documents, please answer this question: {query} 
Documents: {chr(10).join([f'- {doc.page_content}' for doc in relevent_doc])}

please provide a cleear, helpful answer using only the information from these documents. If you can't find the answer then reply with more detaiuls needed
"""

model=ChatGroq(model="llama-3.3-70b-versatile")
message=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input),
]

result=model.invoke(message)

print("--Generated response--")

print("content only:")
print(result.content)

