from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage


load_dotenv()

persist_directory="db/chromadb"

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db=Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space':'cosine'}
)

# query="types of machune learning"

# retriver=db.as_retriver(search_kwargs={'k':3})

# relevent_docs=retriver.invoke(query)

# print(f"use query : {query}")
# print(f"-------context-----")


# for i, doc in enumerate(relevent_docs,1): 
#     print(f"Document  {i}: \n {doc.page_content}\n")

# combined_input= f"""Based on the following documents, please answer this question: {query} 
# Documents: {chr(10).join([f'- {doc.page_content}' for doc in relevent_doc])}

# please provide a cleear, helpful answer using only the information from these documents. If you can't find the answer then reply with more detaiuls needed
# """


model=ChatGroq(model="llama-3.3-70b-versatile")

# message=[
#     SystemMessage("you're an helpful assistant"),
#     HumanMessage(combined_input)


# ]
# response=model.invoke(message)
# print(response.content)


chat_history=[]

def ask_question(user_question): 
    print("__you asked:",user_question)

    if chat_history: 
        message=[
            SystemMessage(content="Given the chat history,rewrite the new question to be standalone and searchable. ")
        ]+chat_history+[HumanMessage(content=f"New qeustion: {user_question}")]
        
        result=model.invoke(message)
        search_question=result.content.strip()
        print(f"searching for: {search_question}")
    
    else: 
        search_question=user_question
    
    retriver=db.as_retriever(search_kwargs={'k':3})
    docs=retriver.invoke(search_question)

    print(f"Found {len(docs)} relevent documents")
    for i,doc in enumerate(docs,1): 
        lines=doc.page_content.split("\n")[:2]
        preview='\n'.join(lines)
        print(f"Doc {i}: {preview}")
    
    combined_input= f"""Based on the following documents, please answer this question: {search_question} 
    Documents: {chr(10).join([f'- {doc.page_content}' for doc in docs])}
    please provide a cleear, helpful answer using only the information from these documents. If you can't find the answer then reply with more detaiuls needed
    """

    # model=ChatGroq(model="llama-3.3-70b-versatile")

    message=[
        SystemMessage("you're an helpful assistant"),
        HumanMessage(combined_input)


    ]
    response=model.invoke(message)
    answer=response.content 

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"answer:{answer}")
    return answer







def start_chat(): 
    print("ask me a question! type quit to exit")

    while True: 
        question=input("\n your question: ")
        if question.lower()=='quit': 
            print("bye")
            break 
        ask_question(question)


if __name__=='__main__': 
    start_chat()

