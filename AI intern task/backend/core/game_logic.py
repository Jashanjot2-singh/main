from core.ai_client import check_guess_with_ai
from core.moderation import is_clean

# In-memory storage
sessions = {}  # session_id: {seed, guesses: [], score}
verdict_cache = {}  # (seed, guess): "yes"/"no"
global_guess_counter = {}  # guess: int

def handle_guess(session_id, guess, persona):
    if not is_clean(guess):
        return {"result": "error", "message": "Inappropriate content"}

    session = sessions.setdefault(session_id, {
        "seed": "rock",
        "guesses": [],
        "score": 0
    })

    seed = session["seed"]

    if guess in session["guesses"]:
        return {
            "result": "duplicate",
            "score": session["score"],
            "message": "Duplicate guess. Game over.",
            "last_guesses": session["guesses"][-5:],
            "global_count": global_guess_counter.get(guess, 0)
        }

    cache_key = (seed, guess)
    if cache_key in verdict_cache:
        verdict = verdict_cache[cache_key]
    else:
        verdict = "yes" if check_guess_with_ai(seed, guess, persona) else "no"
        verdict_cache[cache_key] = verdict

    if verdict == "no":
        return {
            "result": "incorrect",
            "score": session["score"],
            "message": f"{guess} does not beat {seed}.",
            "last_guesses": session["guesses"][-5:],
            "global_count": global_guess_counter.get(guess, 0)
        }

    # Valid & new guess
    session["guesses"].append(guess)
    session["score"] += 1
    global_guess_counter[guess] = global_guess_counter.get(guess, 0) + 1

    return {
        "result": "correct",
        "score": session["score"],
        "message": f"✅ Nice! {guess} beats {seed}. Guessed {global_guess_counter[guess]} times globally.",
        "last_guesses": session["guesses"][-5:],
        "global_count": global_guess_counter[guess]
    }
