# AI Agent with RAG (OpenAI + Qdrant)
## Overview

This project is a simple AI assistant for ABC Company that can answer normal questions directly, and when needed it goes and searches company documents (policies, FAQs, SOPs) before answering.

### Idea is simple:

- If question is general → answer directly using OpenAI

- If question needs company knowledge → fetch data from documents using Qdrant and then answer

This was built mainly for an AI Agent + RAG assignment, so focus is on clarity and correct flow, not over-engineering.

## What this system can do

- Accept user questions via API

- Decide on its own:

    - answer directly

    - OR call a tool to fetch documents

- Search company PDFs using vector search (Qdrant)

- Answer in a clean, structured way

- Keep basic chat memory using session id

- Show document sources when RAG is used

- Simple Streamlit UI to chat with the agent

## Architecture (High level)
``` 
User (Streamlit UI)
        |
        v
FastAPI (/ask endpoint)
        |
        v
OpenAI Chat Model (Agent)
   |           |
   |           |
No tool     Tool call
(answer)     |
              v
        Qdrant Vector DB
              |
              v
        Relevant Chunks
              |
              v
        OpenAI (final answer)
```
## Tech Stack Used

- Python

- FastAPI – backend API

- OpenAI API – chat completion & agent reasoning

- Qdrant (Cloud) – vector database

- Ollama (local) – embeddings using nomic-embed-text

- LangChain – PDF loading & chunking

- Streamlit – simple chat UI

- Requests – calling Ollama locally

## Documents Used

I created 5 sample company PDFs for ABC Company:

- Leave & Attendance Policy (name as docs_1)

- Travel & Expense Policy (name as docs_2)

- Information Security Policy (name as docs_3)

- Code of Conduct (name as docs_4)

- Product FAQ & Support SOP (name as docs_5)

These are just sample docs, no real company data.

## How RAG Works Here (Simple explanation)

1) PDFs are loaded using LangChain PyPDFLoader

2) Text is split into small chunks (size 500, overlap adjusted)

3) Each chunk is embedded using nomic-embed-text via Ollama

4) Embeddings are stored in Qdrant Cloud

5) When a user asks something policy related:
   - Query is embedded
    - Top 5 similar chunks are fetched from Qdrant
    - These chunks are sent to OpenAI as context

6) Model answers based only on retrieved content

## AI Agent Logic (Important part)

The agent uses **Tool calling**.

Rules given in system prompt:

- If user asks about:
    - leave policy
    - expenses
    - security
    - SOPs
    - or says “according to policy”
      - → call Qdrant tool

- If user asks general stuff (hi, hello, what can you do)
  - → answer directly

- If not sure → better to call tool

So the model itself decides when to use retrieval.

## API Design
Endpoint

**POST /ask**

```
Request
{
  "query": "What is the leave carry forward policy?",
  "session_id": "optional"
}
```
**Response**
```
{
  "answer": "structured answer from assistant",
  "source": ["doc1", "doc2"]
}
```
If no documents were used, source list will be empty.

## Session Memory

- Memory is session based

- session_id keeps last few messages

- Stored in memory (not DB)

- Enough for assignment and demo

## Running Locally
1. Start Ollama
ollama pull nomic-embed-text
ollama serve
2. Set environment variables
export OPENAI_API_KEY="your_openai_key"
export QDRANT_URL="https://your-qdrant-host:6333"
export QDRANT_API_KEY="your_qdrant_key"
3. Run FastAPI
uvicorn app:app --reload
4. Run Streamlit UI
streamlit run streamlit_app.py

Open browser at http://localhost:8501

## Why I chose this approach

- Tool calling makes agent behavior very clear

- Qdrant is fast and simple for vector search

- Ollama embeddings avoid extra API cost

- FastAPI is clean and production friendly

- Streamlit is easiest way to demo chat UI

- No unnecessary complexity added

## Limitations

- Session memory is in-memory only

- No authentication

- No reranking of results

- Ollama must be running locally

- Not optimized for very large documents

## Future Improvements

- Persistent memory (Redis / DB)

- Reranking with cross-encoder

- Streaming responses

- Dockerized deployment

- Azure App Service deployment

- Better document citations (page level)

## Final Note

This project is intentionally kept simple and readable.
Main goal was to show:

- agent reasoning
- tool calling
- RAG flow
- clean API design