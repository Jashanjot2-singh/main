# Rock–Paper–Scissors–Plus (Google ADK Agent)

This repo is a small Google ADK agent that plays **Rock–Paper–Scissors–Plus** for **exactly 3 rounds**. In addition to rock/paper/scissors, there’s a special move **bomb** that each player can use **only once per game**. Bomb beats everything, and bomb vs bomb is a draw. Invalid input still consumes the round.

## How to run

1) Install ADK
```bash
pip install -U google-adk
```
2) Put the agent code in:
```bash
my_agent/agent.py
```
3) Add your API key in:
```bash
my_agent/.env
GOOGLE_API_KEY="YOUR_KEY"
```
4) Run:
```bash
adk run my_agent
```

Type hi to start, then play rock, paper, scissors, or bomb.

## State model (what I store and why)

I keep game state in ADK session state so it persists across turns (instead of relying on prompt memory).

#### Core state keys
1) intro_shown → ensures rules are shown only once (prevents the “intro loop”)
2) round → current round number (1..3)
3) user_score, bot_score → score tracking across rounds
4) user_bomb_used, bot_bomb_used → enforces “bomb once per player”
5) history → list of per-round records (user move, bot move, outcome). Helpful for debugging and final summaries.
#### Fairness / commit–reveal keys
1) bot_move, bot_nonce, bot_commit_hash
- Before the user plays a round, the bot commits to its move by publishing a SHA-256 hash of nonce:move.
- After the user plays, the bot reveals move + nonce so anyone can verify the hash matches.

#### CLI reliability key (ADK-specific)

1) temp:direct_reply
- This holds the exact message to show to the user so the CLI output stays reliable even when the “post-tool” model message is empty.

## Agent + Tool design

### One agent, one explicit tool.

1) The agent is a simple coordinator: it always calls one tool, game_step(user_message, tool_context).
2) The tool is the referee. It does all deterministic logic:
- parse input
- validate moves (including “bomb only once”)
- apply win rules
- update state (round/score/bomb flags/history)
- end automatically after round 3
- generate the exact user-facing message

I kept the “game logic” inside the tool on purpose. The LLM is great at conversation, but rules + scoring should be deterministic so it never “hallucinates” a winner or forgets a bomb usage.

## How I kept it fair

I used a commit–reveal approach:
- Bot picks its move and a random nonce
- Bot shows the user a hash commit (like a sealed envelope)
- User plays
- Bot reveals the move + nonce so the user can confirm the commit wasn’t changed

This prevents the bot from “choosing after seeing your move” and makes the game feel trustworthy.

## Tradeoffs I made

- I kept the agent extremely strict (tool-driven) rather than letting the LLM generate free-form text. That reduces flexibility, but improves correctness.

- Parsing is intentionally simple (keyword-based). It’s robust enough for normal chat (“I choose rock!!”), but not a full NLP pipeline.

- The CLI reliability hook (temp:direct_reply + callback) is a pragmatic workaround to ensure the user always sees responses in adk run. It’s not strictly game logic, but it makes the experience stable.

## What I’d improve with more time

- Add unit tests for all edge cases (invalid inputs, bomb reuse attempts, bomb vs bomb, repeated restarts).
- Improve NLU (handle more phrasing, multiple moves in one message, clearer error messages).
- Add a nicer round summary at the end (show all rounds neatly from history).
- Add optional “strict mode” vs “friendly mode” (strict consumes invalid rounds exactly as spec; friendly could ask for re-entry while still recording that the round was wasted).