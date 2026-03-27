from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

alibaba_text=""" Alibaba currently offers extensive free resources to explore their Qwen AI models (similar to OpenAI/ChatGPT): 
Alibaba Cloud
Free Tokens: New users can get up to 70+ million free AI tokens.
Model Studio: You can start for free, with costs only incurred when you invoke models after the free quota is exhausted.
Free Trial Duration: Some trials are available for up to 12 months for various products.
Key Requirement: Requires setting up an Alibaba Cloud account and passing real-name registration."""


semantic_chunker=SemanticChunker(
    embeddings=HuggingFaceEmbeddings(),
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=70

)
chunks=semantic_chunker.split_text(alibaba_text)

print("Semantic chunks")

for i, chunk in enumerate(chunks,1): 
    print(f"{i} : {len(chunk)} chars ")
    print(f"{chunk}")
    print("\n")