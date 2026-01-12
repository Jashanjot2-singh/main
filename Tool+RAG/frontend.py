import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="ABC Company Chatbot", layout="centered")
st.title("🤖 ABC Company Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask a question about ABC company policies, FAQs, or anything else...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call FastAPI backend
    try:
        response = requests.post(
            API_URL,
            json={
                "query": user_input,
                "session_id": st.session_state.session_id,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "")
        sources = data.get("source", [])

    except Exception as e:
        answer = f" Error contacting backend: {e}"
        sources = []

    # Build assistant message
    assistant_text = answer
    if sources:
        assistant_text += "\n\n**Sources:**\n"
        for s in sources:
            assistant_text += f"- {s}\n"

    # Show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
