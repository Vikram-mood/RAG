from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

alibaba_text=""" Alibaba currently offers extensive free resources to explore their Qwen AI models (similar to OpenAI/ChatGPT): 
Alibaba Cloud
Free Tokens: New users can get up to 70+ million free AI tokens.
Model Studio: You can start for free, with costs only incurred when you invoke models after the free quota is exhausted.
Free Trial Duration: Some trials are available for up to 12 months for various products.
Key Requirement: Requires setting up an Alibaba Cloud account and passing real-name registration."""

splitter1=CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20
)
chunks1=splitter1.split_text(alibaba_text)
for i,chunk in enumerate(chunks1,1): 
    print(f"chunk {i}: ({len(chunk)})chars")
    print(f"{chunk}")
    print("\n")



splitter2=RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"," ","."],
    chunk_size=100,
    chunk_overlap=0
)
chunks2=splitter2.split_text(alibaba_text)

for i, chunk in enumerate(chunks2,1): 
    print(f"chunk {i}: ({len(chunk)})chars")
    print(f"{chunk}")
    print("\n")

