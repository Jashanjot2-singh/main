from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.game_logic import handle_guess

app = FastAPI()

class GuessRequest(BaseModel):
    session_id: str
    guess: str
    persona: str = "cheery"

@app.post("/guess")
def guess_endpoint(req: GuessRequest):
    result = handle_guess(req.session_id, req.guess.strip().lower(), req.persona)
    if result["result"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result
