# 📘 Resume Screening Tool – Architecture Overview

This document explains the full system architecture of the AI-powered Resume Screening Tool using Node.js + React + RAG (Retrieval-Augmented Generation).

## 🔷 1. High-Level Overview

The system consists of:

- **Frontend (React + TypeScript)**
- **Backend (Node.js + Express + TypeScript)**
- **Gemini 2.5 Flash** → Used for match analysis, question answering, and RAG final generation
- **Gemini Embeddings 3072-D** → Used to create embeddings for resume chunks
- **Qdrant Cloud** → Stores vector embeddings for retrieval
- **Ollama (local)** → Used optionally for embedding fallback (nomic-embed-text)
- **MySQL (XAMPP)** → Stores chat history (returns last 5 messages per session)
- **pdf-parse** → Extracts text from PDF documents

## 🔷 2. System Architecture Flow

**React UI (Upload, Analysis, Chat)**
- → HTTP Requests
- → **Node.js API (Express + TypeScript)**
- → **Document Processing Layer (pdf-parse → text chunks)**
- → **Gemini Embeddings 3072D (Create vector embeddings)**
- → **Qdrant Vector DB (Store resume embeddings)**
- → **RAG Retrieval (Search relevant chunks)**
- → **Gemini 2.5 Flash LLM (Generates final answers)**
- → **MySQL XAMPP (Save last 5 chat history)**
- → **Response to UI**

## 🔷 3. Backend Architecture

### 3.1 Directory Structure
```
backend/
├── src/
│   ├── routes/
│   │   ├── analyze.ts
│   │   └── chat.ts
│   ├── services/
│   │   ├── gemini.ts
│   │   ├── embeddings.ts
│   │   ├── vectorDB.ts
│   │   ├── rag.ts
│   │   └── mysql.ts
│   ├── utils/pdfParser.ts
│   └── index.ts
```

## 🔷 4. Document Processing Pipeline

### 4.1 PDF → Text Extraction

**pdf-parse** is used to extract plain text from resume and job description.

### 4.2 Chunking Strategy

- **Chunk size:** 500 words
- **Overlap:** 50 words

Chunking improves:
- semantic search,
- retrieval accuracy,
- RAG context relevance.

## 🔷 5. Embeddings Architecture

Your updated architecture uses:

✅ **Gemini Embeddings 3072-Dimensional Vectors**

The embedding model used:
```
gemini-embedding-2.0  (3072 dimensions)
```

### Why 3072?

- Larger vector space
- Higher semantic richness
- Better matching for resumes with mixed structure
- Ideal for skills extraction & technical document retrieval

### Flow:

- resume chunk → Gemini Embeddings (3072D) → Qdrant
- question → Gemini Embeddings (3072D) → Qdrant search

## 🔷 6. Vector Database Architecture (Qdrant)

Each session creates one collection or uses a common collection with filtering.

### Stored fields:

| Field | Description |
|-------|-------------|
| id | UUID of chunk |
| text | Resume chunk text |
| embedding | 3072-D vector |
| session_id | Group by user session |

### Distance Metric

**cosine**

## 🔷 7. RAG Architecture

### 7.1 Retrieval Flow

1. Convert user question → 3072-D embedding (Gemini Embedding model)
2. Perform a vector similarity search in Qdrant
3. Retrieve top K relevant chunks (K=5)
4. Build context:
   - [Retrieved Resume Chunks]
   - + Last 5 chat messages (MySQL)
   - + Job Description (optional)
5. Invoke Gemini 2.5 Flash with structured system prompt
6. Generate final answer

## 🔷 8. Chat History Layer (MySQL)

### Table Structure
```sql
id INT AUTO_INCREMENT PRIMARY KEY
session_id VARCHAR(255)
role ENUM('user','assistant')
message TEXT
created_at TIMESTAMP
```

### Rules:

- Store every message
- Always retrieve only last 5 messages
- Delete old messages periodically

## 🔷 9. Resume Match Analysis Architecture

For `/api/analyze`:

1. Extract resume & JD text
2. Send both to Gemini 2.5 Flash
3. Gemini returns structured JSON:
```json
   {
     "score": 75,
     "strengths": [...],
     "gaps": [...],
     "insights": [...]
   }
```
4. Resume text is chunked → embedded → stored in Qdrant
5. Session ID returned to frontend

## 🔷 10. Frontend Architecture

Built using:
- React + TypeScript
- Vite
- Tailwind CSS
- shadcn/ui

### Modules:
- `UploadSection.tsx`
- `AnalysisResults.tsx`
- `ChatInterface.tsx`

## 🔷 11. RAG vs NON-RAG Behavior

| Feature | Without RAG | With RAG |
|---------|-------------|----------|
| Uses entire resume? | Yes (BAD) | No |
| Retrieval? | ❌ | ✔️ |
| Embeddings? | ❌ | ✔️ Gemini 3072D |
| Vector Search | ❌ | ✔️ Qdrant |
| Accuracy | Low | Very High |
| Cost | More (large input) | Less |

Your implementation uses true RAG — correct and required.

## 🔷 12. Component Flow

**Frontend**
- → REST API Calls
- → **Express API**
  - → **PDF Parser**
  - → **Gemini API**
  - → **Embedding Service**
- → **Resume Analyzer**
  - → **Qdrant Vector DB**
  - → **MySQL Storage**

## 🔷 13. Key Highlights

- ✔ Uses Gemini Embedding 3072D
- ✔ True RAG: embedding → search → retrieval → LLM
- ✔ Qdrant Cloud with HTTPS + API Key
- ✔ MySQL chat memory (last 5 messages only)
- ✔ Session-based isolation
- ✔ Full end-to-end AI workflow

## 🔷 14. Future Enhancements

- Add Redis cache for faster retrieval
- Add multi-resume comparison
- Add user authentication (JWT)
- Add queueing for large files