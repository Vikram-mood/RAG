# RAG (Retrieval-Augmented Generation) Explorations

A comprehensive repository for building and exploring Retrieval-Augmented Generation (RAG) systems using **LangChain**, **ChromaDB**, **HuggingFace**, and **Groq (Llama-3)**. 

This project covers everything from basic document ingestion and text splitting methodologies to history-aware conversational retrieval and complex multi-modal RAG pipelines.

## 🚀 Features

- **Document Ingestion (`ingestion_pipeline.py`)**: End-to-end pipeline to load `.txt` and `.pdf` files from your local data directories, split them into chunks, generate embeddings via HuggingFace, and persist them into a local Chroma vector database.
- **History-Aware Chat (`history_aware_rag.py`)**: An interactive command-line application that acts as a Chatbot. It intelligently rewrites user queries based on conversation history before retrieving context from ChromaDB, allowing for robust multi-turn conversations.
- **Text Splitting Experiments (`textsplitters.py`, `semantic_chunking.py`)**: Scripts demonstrating various ways to divide documents to improve retrieval accuracy, including Character splitting, Recursive Character splitting, and Semantic chunking.
- **Retrieval Pipeline (`retreveal_pipeline.py`)**: Implementation of basic document retrieval mechanisms connecting the vector store with language models to produce grounded answers.
- **Multi-Modal RAG (`multi_modal_rag.ipynb`)**: An advanced Jupyter Notebook handling complex PDF documents. This notebook extracts not just raw text, but also parses tables and processes images (using libraries like `unstructured` and Pillow) to augment LLM answers with visual context.

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://python.langchain.com/) for orchestrating the RAG flows.
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Local persistent storage).
- **Embeddings**: [HuggingFace Embeddings](https://huggingface.co/) (`all-MiniLM-L6-v2` and others) for dense vector representations.
- **LLM Engine**: [Groq](https://groq.com/) using `llama-3.3-70b-versatile` for blazing-fast generation.
- **Document Processing**: `unstructured` for parsing complex PDFs containing multiple modalities (images, tables, text).

## 📦 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vikram-mood/RAG.git
   cd RAG
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On MacOS/Linux
   # OR: .\venv\Scripts\activate on Windows
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(For the Multi-Modal pipeline, ensure you have system dependencies like `poppler` and `tesseract` installed, as well as `unstructured[all-docs]`.)*

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys.
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 🏃‍♂️ Usage Examples

### 1. Ingest Documents into Vector DB
Drop your `.txt` or `.pdf` files into `data/text_files/` and run the ingestion script to create a persistent ChromaDB store.
```bash
python ingestion_pipeline.py
```

### 2. Run the Conversational RAG Chat
Start an interactive console shell to query the ingested documents with conversation memory.
```bash
python history_aware_rag.py
```

### 3. Explore the Multi-Modal Capabilities
Launch Jupyter to explore the notebook designed for reading text, images, and tables.
```bash
jupyter notebook multi_modal_rag.ipynb
```

---
*Developed by [Vikram](https://github.com/Vikram-mood/)*
