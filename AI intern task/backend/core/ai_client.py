import requests

def check_guess_with_ai(seed, guess, persona):
    system_prompt = f"You are a {'cheerful' if persona == 'cheery' else 'serious'} game host. Answer 'Yes' or 'No' only."
    user_prompt = f"Does '{guess}' beat '{seed}'?"

    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post("http://localhost:11434/api/chat", json=payload)
        answer = res.json()['message']['content'].strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print("Error querying Ollama:", e)
        return False
