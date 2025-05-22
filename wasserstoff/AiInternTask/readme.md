# 📑 RAG Chatbot + OCR for Document Understanding

## Overview

This project is part of the **Wasserstoff Gen-AI Internship Task**. It implements a **Retrieval-Augmented Generation (RAG)** chatbot capable of performing intelligent search and question-answering over various document formats, including scanned images using OCR.

### 🎯 Objective

- Ingest and manage 75+ documents (PDF, TXT, DOCX, Excel, PNG, JPG).
- Perform OCR on scanned documents.
- Store and index documents using semantic embeddings in a vector database.
- Answer natural language queries with citation-level precision.
- Identify common themes across documents.
- Present document-level citation mapping.

---

## 🔧 Tech Stack and Design

### 🔤 OCR: TrOCR Fine-Tuned on IAM Dataset
- **Model:** `microsoft/trocr-base-handwritten`
- **Why TrOCR?** TrOCR provides accurate handwritten text recognition. I further **fine-tuned** it on the **IAM dataset** (11,000 images from 700 writers) for domain-specific generalization.
- Used for scanned image extraction (PNG, JPG, JPEG).

### 🧠 Language Model: `llama3.2:1b` via Ollama
- **Why LLAMA3.2:1b?** Lightweight and efficient, hosted locally via [Ollama](https://ollama.com/), making it a cost-effective, fast inference choice for local development and deployment.
- Used for contextual generation and final answers.

### 🧬 Embedding Model: `nomic-embed-text`
- **Why Nomic Embed?** High-quality open-source text embeddings with strong performance in semantic search. Hosted via local Ollama API.
- Converts text chunks into vector embeddings for retrieval.

### 🧠 Vector Database: Qdrant
- **Why Qdrant?** A fast, production-grade vector search engine that enables efficient similarity search.
- Stores and indexes document embeddings for real-time querying.

### 🧰 Framework: Streamlit
- **Why Streamlit?** Provides a rapid UI for data science and LLM applications with minimal boilerplate.
- Used for file uploads, text input, result display, and debugging.

---

## 📂 Features

- Upload files in multiple formats: `pdf`, `txt`, `docx`, `xlsx`, `png`, `jpg`, `jpeg`.
- Preprocessing pipeline with OCR and document parsing.
- Chunking and embedding of document content.
- Creation and querying of a Qdrant vector database.
- RAG query answering with context-aware generation.
- Streamlit-based visualization with query results and citations.

---

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
### 2. Run the Streamlit app
```bash
streamlit run app.py
```
#### Make sure the following services are running:

1. Ollama for llama3.2:1b and nomic-embed-text

2. Qdrant server (default at localhost:6333)

### 🧠 Internship Task Highlights
This implementation covers core objectives from the Wasserstoff Gen-AI Internship Task:

✅ Document ingestion and OCR

✅ Query processing with context

✅ Document-level theme analysis

✅ Citation support (basic level)

✅ Extensible design for sentence-level citation and visual linking

### 📬 Contact
For further questions regarding the project, please contact:

Jashanjot singh

### 🔗 Useful Links
1. Ollama (LLMs + Embeddings)

2. Qdrant Vector DB

3. IAM Dataset Info

### 💡 Future Improvements
1. Sentence/paragraph-level citations

2. Theme clustering visualization

3. Multi-document cross-context analysis

4. FastAPI backend for production-grade deployment

