# 🧠 What Beats Rock – Generative AI Guessing Game

This is a Generative AI-based interactive word guessing game inspired by the "What Beats Rock?" concept. It uses FastAPI for the backend, Streamlit for the frontend, and a locally hosted Ollama model (LLaMA3) to evaluate player guesses in real-time.

---

## 🚀 Project Overview

- **Seed Word:** The game starts with a fixed word – `"rock"`.
- **Goal:** The user submits guesses they think "beat" the seed.
- **Validation:** A local AI model (LLaMA3 via Ollama) determines whether the guess beats the seed.
- **Correct Guess:** The guess is added to the session’s history, score increases.
- **Duplicate Guess:** Triggers **Game Over**.
- **Wrong Guess:** No score change.
The game tracks:
- Session score
- Guess history (last 5)
- Global guess count (simulated in memory)
- Animated feedback via Streamlit

---

## ❗️Why No MongoDB or Redis?

Due to an issue on my local machine, I was **unable to run MongoDB or Redis successfully**. I attempted multiple solutions:
- Using `mongod` manually
- Installing via Homebrew/Windows installer
- Verifying ports and services

Despite these, I ran into `ECONNREFUSED` errors on `localhost:27017` and wasn't able to resolve it in time due to a **submission deadline** and overlapping **college semester exams**.

To ensure a working demo and complete submission, I refactored the backend logic to use **in-memory data structures** (`dict`s) instead of external databases or caches.

---

## 💻 Tech Stack

| Component     | Tool Used     |
|---------------|---------------|
| Backend API   | FastAPI       |
| Frontend UI   | Streamlit     |
| AI Model      | Ollama (LLaMA3) |
| Data Storage  | Python `dict` (in-memory) |
| Verdict Cache | Python `dict` (in-memory) |
| Deployment    | Localhost only |

---

## 📁 Folder Structure

genai-intern-game/ ├── backend/ │ ├── core/ │ │ ├── ai_client.py # Talks to Ollama │ │ ├── game_logic.py # Game state in memory │ │ └── moderation.py # Profanity filter │ ├── main.py # FastAPI entry point │ └── requirements.txt │ ├── frontend/ │ ├── streamlit_app.py # Streamlit app │ └── requirements.txt

yaml
Copy code

---

## 🛠 Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Ollama installed and running:

```bash
ollama run llama3
This exposes the AI model at: http://localhost:11434
```
2. Run the Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
3. Run the Frontend (Streamlit)
```bash
cd ../frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```
4. Play the Game 🎮
Open browser: http://localhost:8501

Enter guesses like "paper", "water", "lava"...

Watch your score increase with creative inputs!

## 🔒 Notes on Limitations

No persistent data storage: All data is lost when the app restarts.
Not scalable for production, as everything is stored in memory.
Only one user per session, as session IDs are stored client-side (in Streamlit).

## ✅ Future Improvements
Add MongoDB + Redis back when working
Docker support for one-click setup
Live deployment via Render or Railway
Auth, multiplayer support, and game reset feature

## 📧 Contact
For queries, contact:
#### Name: Jashanjot singh
#### Email: Jashanjotdhiman@gmail.com
#### Role Applied: AI Software Intern (Wasserstoff)

