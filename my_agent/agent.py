# agent.py
from __future__ import annotations

import hashlib
import random
import re
import secrets
from typing import Any, Dict, Optional, Tuple

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import ToolContext
from google.genai import types  # dependency installed with google-adk

MOVES = ("rock", "paper", "scissors", "bomb")
_SENTINEL = object()


# ---------------------------
# CLI output reliability fix:
# ---------------------------
# ADK sometimes calls the model after a tool result to produce final assistant text.
# We store the exact reply in state and return it here as a valid LlmResponse.
def direct_reply_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    msg = callback_context.state.get("temp:direct_reply", None)
    if msg:
        callback_context.state["temp:direct_reply"] = None  # clear so it won't repeat
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=str(msg))])
        )
    return None


def _set_direct_reply(tool_context: ToolContext, msg: str) -> None:
    tool_context.state["temp:direct_reply"] = msg


def _ensure_key(state: Any, key: str, default_value: Any) -> None:
    cur = state.get(key, _SENTINEL)
    if cur is _SENTINEL:
        state[key] = default_value


def _init_state(state: Any) -> None:
    _ensure_key(state, "intro_shown", False)

    _ensure_key(state, "round", 1)
    _ensure_key(state, "user_score", 0)
    _ensure_key(state, "bot_score", 0)

    _ensure_key(state, "user_bomb_used", False)
    _ensure_key(state, "bot_bomb_used", False)

    _ensure_key(state, "bot_move", None)
    _ensure_key(state, "bot_nonce", None)
    _ensure_key(state, "bot_commit_hash", None)

    _ensure_key(state, "game_over", False)
    _ensure_key(state, "history", [])


def _reset_state(state: Any) -> None:
    state["intro_shown"] = False

    state["round"] = 1
    state["user_score"] = 0
    state["bot_score"] = 0

    state["user_bomb_used"] = False
    state["bot_bomb_used"] = False

    state["bot_move"] = None
    state["bot_nonce"] = None
    state["bot_commit_hash"] = None

    state["game_over"] = False
    state["history"] = []

    state["temp:direct_reply"] = None


def _rules_text_5_lines() -> str:
    return "\n".join(
        [
            "Rules: Best of 3 rounds. Moves: rock/paper/scissors + bomb (once per player).",
            "bomb beats any other move; bomb vs bomb is a draw.",
            "Invalid input wastes the round (still counts).",
            "Fair-play: I commit my move (hash) before you play, then reveal after.",
            "Type your move now.",
        ]
    )


def _is_restart_intent(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in ["new game", "restart", "reset", "play again"])


def _pick_bot_move(state: Any) -> str:
    choices = ["rock", "paper", "scissors"]
    if not state.get("bot_bomb_used", False):
        choices.append("bomb")

    # bomb is rarer
    if "bomb" in choices:
        return random.choices(choices, weights=[1.0, 1.0, 1.0, 0.2], k=1)[0]
    return random.choice(choices)


def _ensure_commit_for_round(state: Any) -> None:
    if state.get("bot_commit_hash") and state.get("bot_move") and state.get("bot_nonce"):
        return

    bot_move = _pick_bot_move(state)
    nonce = secrets.token_hex(8)
    commit = hashlib.sha256(f"{nonce}:{bot_move}".encode("utf-8")).hexdigest()

    state["bot_move"] = bot_move
    state["bot_nonce"] = nonce
    state["bot_commit_hash"] = commit


def _extract_user_move(user_message: str) -> Optional[str]:
    t = user_message.lower()
    t = re.sub(r"[^a-z\s]", " ", t)

    synonyms = {
        "bomb": ["bomb", "nuke", "grenade"],
        "rock": ["rock", "stone"],
        "paper": ["paper"],
        "scissors": ["scissors", "scissor"],
    }

    best: Optional[Tuple[int, str]] = None
    for move, words in synonyms.items():
        for w in words:
            idx = t.find(w)
            if idx != -1:
                if best is None or idx < best[0]:
                    best = (idx, move)
    return best[1] if best else None


def _validate_move(move: Optional[str], state: Any) -> Tuple[bool, str]:
    if move is None:
        return False, "I couldn't find a valid move in that message."
    if move not in MOVES:
        return False, f"'{move}' is not a valid move."
    if move == "bomb" and state.get("user_bomb_used", False):
        return False, "You already used bomb earlier. Bomb can be used only once per game."
    return True, "OK"


def _resolve(user_move: str, bot_move: str) -> Tuple[str, str]:
    if user_move == "bomb" and bot_move == "bomb":
        return "draw", "bomb vs bomb → draw."
    if user_move == bot_move:
        return "draw", "Same move → draw."

    if user_move == "bomb":
        return "user", "bomb beats everything."
    if bot_move == "bomb":
        return "bot", "bomb beats everything."

    beats = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    if beats[user_move] == bot_move:
        return "user", f"{user_move} beats {bot_move}."
    return "bot", f"{bot_move} beats {user_move}."


def _final_message(state: Any, last_round=None) -> str:
    user = int(state.get("user_score", 0))
    bot = int(state.get("bot_score", 0))

    if user > bot:
        winner = "YOU WIN 🎉"
    elif bot > user:
        winner = "BOT WINS 🤖"
    else:
        winner = "DRAW 🤝"

    lines = []
    if last_round:
        r, um, bm, out, expl, commit, nonce = last_round
        lines.append(
            f"Round {r}/3\nYou: {um} | Bot: {bm} | {out.upper()} ({expl})\n"
            f"Commit reveal: {commit} with nonce={nonce}\n"
        )

    lines.append(f"Final Score: You {user} - Bot {bot}")
    lines.append(f"Final Result: {winner}")
    lines.append("Type 'new game' to play again.")
    return "\n".join(lines)


def game_step(user_message: str, tool_context: ToolContext) -> Dict[str, Any]:
    state = tool_context.state
    _init_state(state)

    # Restart if requested
    if _is_restart_intent(user_message):
        _reset_state(state)

    # If game ended: only restart allowed
    if state.get("game_over", False):
        msg = "Game is already over. Type 'new game' to start again."
        _set_direct_reply(tool_context, msg)
        return {"status": "success", "message": msg}

    # Show intro exactly once (DO NOT loop)
    if not state.get("intro_shown", False):
        state["intro_shown"] = True
        _ensure_commit_for_round(state)
        msg = _rules_text_5_lines() + f"\n\nRound 1/3\nMy commit: {state['bot_commit_hash']}\nYour move?"
        _set_direct_reply(tool_context, msg)
        return {"status": "success", "message": msg}

    # ---- From here on, every user message is treated as a round input ----
    round_no = int(state.get("round", 1))
    if round_no > 3:
        state["game_over"] = True
        msg = "Game is over. Type 'new game' to restart."
        _set_direct_reply(tool_context, msg)
        return {"status": "success", "message": msg}

    _ensure_commit_for_round(state)

    user_move = _extract_user_move(user_message)
    valid, reason = _validate_move(user_move, state)

    bot_move = state.get("bot_move")
    bot_nonce = state.get("bot_nonce")
    bot_commit = state.get("bot_commit_hash")

    # Invalid input wastes the round
    if not valid:
        state["history"].append(
            {"round": round_no, "user_move": None, "bot_move": bot_move, "outcome": "wasted", "note": reason}
        )
        state["round"] = round_no + 1

        # Clear commit for next round
        state["bot_move"] = None
        state["bot_nonce"] = None
        state["bot_commit_hash"] = None

        if int(state["round"]) > 3:
            state["game_over"] = True
            msg = _final_message(state)
            _set_direct_reply(tool_context, msg)
            return {"status": "success", "message": msg}

        _ensure_commit_for_round(state)
        msg = (
            f"Round {round_no}/3 (WASTED)\n"
            f"Your input was invalid: {reason}\n"
            f"I had committed to: {bot_commit}\n"
            f"Reveal: move={bot_move}, nonce={bot_nonce}\n"
            f"Score: You {state['user_score']} - Bot {state['bot_score']}\n\n"
            f"Round {state['round']}/3\nMy commit: {state['bot_commit_hash']}\nYour move?"
        )
        _set_direct_reply(tool_context, msg)
        return {"status": "success", "message": msg}

    # Track bomb usage
    if user_move == "bomb":
        state["user_bomb_used"] = True
    if bot_move == "bomb":
        state["bot_bomb_used"] = True

    outcome, expl = _resolve(user_move, bot_move)
    if outcome == "user":
        state["user_score"] += 1
    elif outcome == "bot":
        state["bot_score"] += 1

    state["history"].append(
        {"round": round_no, "user_move": user_move, "bot_move": bot_move, "outcome": outcome, "explanation": expl}
    )

    state["round"] = round_no + 1

    # Clear commit for next round
    state["bot_move"] = None
    state["bot_nonce"] = None
    state["bot_commit_hash"] = None

    # End after 3 rounds
    if int(state["round"]) > 3:
        state["game_over"] = True
        msg = _final_message(state, last_round=(round_no, user_move, bot_move, outcome, expl, bot_commit, bot_nonce))
        _set_direct_reply(tool_context, msg)
        return {"status": "success", "message": msg}

    # Next round prompt
    _ensure_commit_for_round(state)
    msg = (
        f"Round {round_no}/3\n"
        f"You played: {user_move}\nBot played: {bot_move}\n"
        f"Result: {outcome.upper()} ({expl})\n"
        f"Commit reveal: {bot_commit} with nonce={bot_nonce}\n"
        f"Score: You {state['user_score']} - Bot {state['bot_score']}\n"
        f"Bombs left: You={'NO' if state['user_bomb_used'] else 'YES'}, "
        f"Bot={'NO' if state['bot_bomb_used'] else 'YES'}\n\n"
        f"Round {state['round']}/3\nMy commit: {state['bot_commit_hash']}\nYour move?"
    )
    _set_direct_reply(tool_context, msg)
    return {"status": "success", "message": msg}


root_agent = Agent(
    model="gemini-2.5-flash",
    name="rps_referee",
    description="Referee for a 3-round Rock–Paper–Scissors–Plus game (with bomb).",
    instruction="Always call the tool `game_step` once per user message.",
    tools=[game_step],
    before_model_callback=direct_reply_before_model,
)
