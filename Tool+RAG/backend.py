import json
import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from openai import OpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")


COLLECTION_NAME = "abc_company_docs"  # MUST be the same collection you created earlier
TOP_K_DEFAULT = 5
VECTOR_SIZE = 768

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

QDRANT_URL = os.getenv("QDRANT_URL", "https://85ea9aea-a625-4e28-8e3e-47db7073a86c.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", )

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL is not set.")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


SESSIONS: Dict[str, List[Dict[str, Any]]] = {}
MAX_TURNS = 12  # keep last ~12 messages per session


def get_session(session_id: str) -> List[Dict[str, Any]]:
    return SESSIONS.setdefault(session_id, [])


def add_to_session(session_id: str, message: Dict[str, Any]) -> None:
    msgs = get_session(session_id)
    msgs.append(message)
    if len(msgs) > MAX_TURNS:
        SESSIONS[session_id] = msgs[-MAX_TURNS:]


def ollama_embed(text: str) -> List[float]:
    """
    Returns a single embedding vector from Ollama.

    Tries:
    1) POST /api/embeddings {model, prompt}
    2) POST /api/embed {model, input:[...]}  (newer)
    """
    text = text.strip()
    if not text:
        return [0.0] * VECTOR_SIZE

    # 1) Older endpoint
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        if r.ok:
            data = r.json()
            vec = data.get("embedding")
            if isinstance(vec, list) and vec:
                return vec
    except Exception:
        pass

    # 2) Newer endpoint
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()

    if "embeddings" in data and data["embeddings"]:
        return data["embeddings"][0]
    if "data" in data and data["data"] and "embedding" in data["data"][0]:
        return data["data"][0]["embedding"]

    raise RuntimeError("Unexpected Ollama embed response format.")


def retrieve_top_chunks(query: str, top_k: int = TOP_K_DEFAULT) -> Tuple[List[Dict[str, Any]], List[str]]:
    vec = ollama_embed(query)
    if len(vec) != VECTOR_SIZE:
        logger.warning("Embedding dim %s != expected %s. Check your Ollama embedding model.", len(vec), VECTOR_SIZE)

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
    )

    chunks: List[Dict[str, Any]] = []
    sources: List[str] = []

    for hit in results:
        payload = hit.payload or {}
        doc_name = payload.get("doc_name") or payload.get("source") or "unknown"
        text = payload.get("text") or payload.get("page_content") or ""
        page = payload.get("page")
        chunk_id = payload.get("chunk_id")

        chunks.append(
            {
                "doc_name": doc_name,
                "score": float(hit.score),
                "page": page,
                "chunk_id": chunk_id,
                "text": text,
            }
        )
        if doc_name not in sources:
            sources.append(doc_name)

    return chunks, sources


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_company_docs",
            "description": "Search ABC Company's internal documents in Qdrant and return the most relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User question to search for."},
                    "top_k": {"type": "integer", "description": "How many chunks to retrieve.", "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]


SYSTEM_PROMPT = """You are an AI assistant for ABC Company.

You have access to a tool `retrieve_company_docs` that searches internal documents (policies, HR, security, expenses, product SOPs).
You must decide whether to answer directly or use the tool.

Decision rules:
- If the question is about ABC Company policies/FAQs/SOPs (leave, attendance, reimbursements, travel, security, IT rules, product support),
  OR the user asks "according to policy/document", you MUST call the tool before answering.
- If the question is general small-talk, greetings, or generic knowledge not tied to ABC documents, answer directly WITHOUT calling tools.
- If uncertain, prefer calling the tool.

Answer style (always):
Return a clear structured response with:
1) Answer
2) Key points
3) Sources (list document names; empty list if no tool used)

Grounding rule:
If you used the tool, base the policy claims ONLY on the retrieved chunks. If the chunks don’t contain the answer, say so and do not invent.
"""


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    source: List[str] = []


app = FastAPI(title="ABC RAG Agent (OpenAI + Qdrant)", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    session_id = req.session_id or str(uuid.uuid4())
    user_query = req.query.strip()

    # Build messages with session memory
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_session(session_id))
    messages.append({"role": "user", "content": user_query})

    # First call: model decides tool vs no tool
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )

    msg = completion.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    final_answer = ""
    sources: List[str] = []

    if tool_calls:
        # Keep the assistant tool-call message in the conversation
        messages.append(msg)

        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments or "{}")

            if fn_name == "retrieve_company_docs":
                top_k = int(fn_args.get("top_k", TOP_K_DEFAULT))
                chunks, srcs = retrieve_top_chunks(fn_args.get("query", user_query), top_k=top_k)
                sources = srcs

                # Compact context for the model
                context_blocks = []
                for c in chunks:
                    hdr = f"[Doc: {c['doc_name']} | Page: {c.get('page')} | Chunk: {c.get('chunk_id')} | Score: {c['score']:.3f}]"
                    context_blocks.append(f"{hdr}\n{c['text']}".strip())

                tool_result = {
                    "top_k": top_k,
                    "sources": sources,
                    "context": "\n\n---\n\n".join(context_blocks),
                }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown tool: {fn_name}"}),
                    }
                )

        # Second call: final grounded answer
        completion2 = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        final_answer = completion2.choices[0].message.content or ""

    else:
        # Direct answer
        final_answer = msg.content or ""
        sources = []

    # Update session memory
    add_to_session(session_id, {"role": "user", "content": user_query})
    add_to_session(session_id, {"role": "assistant", "content": final_answer})

    return AskResponse(answer=final_answer, source=sources)
