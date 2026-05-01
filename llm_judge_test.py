# Author: g023
# License: MIT
# tests llm_judge.py on a difficult problem the model struggles with

from _inc_ollama import (
    llm_stream,
    G_REASONING_MODEL,
)
import re

from llm_judge import llm_judge, JudgeResult

# ----------------------------------------------------------------------
# Helper to display judge results nicely
# ----------------------------------------------------------------------
def show_judge(result: JudgeResult, label: str = "Judge"):
    status = "✅ CORRECT" if result.correct else "❌ INCORRECT"
    print(f"\n  [{label}] {status}")
    print(f"  Confidence : {result.confidence:.2f}")
    print(f"  Matches expected display : {result.matches_expected}")
    print(f"  Reasoning  : {result.reasoning}")

# ----------------------------------------------------------------------
# Helper to extract the right answer text from the training example
# ----------------------------------------------------------------------
def extract_expected(raw: str) -> str:
    """Pull the text between [[right_answer]] and [[/right_answer]]."""
    m = re.search(r"\[\[right_answer\]\](.*?)\[\[/right_answer\]\]", raw, re.DOTALL)
    return m.group(1).strip() if m else raw

# ----------------------------------------------------------------------
# Pre‑defined question and expected answer
# ----------------------------------------------------------------------
raw_example = """
[[question]]
A farmer wants to cross a river and take with him a wolf, a goat and a cabbage. He has a boat with three secure separate compartments. If the wolf and the goat are alone on one shore, the wolf will eat the goat. If the goat and the cabbage are alone on the shore, the goat will eat the cabbage. How can the farmer efficiently bring the wolf, the goat and the cabbage across the river without anything being eaten?
[[/question]]

[[right_answer]]
Place the wolf, goat, and cabbage in separate secure compartments in the boat and row across the river. This will prevent any of them from being eaten by the others.
[[/right_answer]]
"""

question = re.search(r"\[\[question\]\](.*?)\[\[/question\]\]", raw_example, re.DOTALL).group(1).strip()
expected = extract_expected(raw_example)

print("Question:\n", question)
print("\nExpected answer:\n", expected)

CAVEMAN_SYS = (
    "You write Ultra-compressed communication mode. Cuts token usage ~75% by speaking like caveman "
    "while keeping full technical accuracy.\n"
    "Respond terse like smart caveman. All technical substance stay. Only fluff bye. "
    "ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift. Still active if unsure.\n"
    "Drop: articles (a/an/the), filler (just/really/basically/actually/simply), "
    "pleasantries (sure/certainly/of course/happy to), hedging. Fragments OK. "
    "Short synonyms (big not extensive, fix not 'implement a solution for'). "
    "Technical terms exact. Code blocks unchanged. Errors quoted exact.\n"
    "Pattern: [thing] [action] [reason]. [next step].\n"
    "Not: 'Sure! I'd be happy to help with that...' "
    "Drop articles, fragments OK, short synonyms. Classic caveman style. "
    "Maximum classical terseness.\n"
    "Drop caveman when: security warnings, irreversible action confirmations, "
    "multi-step sequences where fragment order risks misread, "
    "compression creates technical ambiguity, user asks to clarify.\n"
    "Resume caveman after clear part done."
)

SYS_PROMPT = """You are an expert reasoner and a master at playing the devils' advocate on any concept.
You identify the single key insight that makes a problem simple.
You carefully check definitions of every term in a question.
You think out complex problems step-by-step. 
You identify and avoid common reasoning traps. 
You verify, validate, reflect, and course correct. 
You argue against common misconceptions.
""" + CAVEMAN_SYS

# ---------------------------------------------
# -------------------------
# Get the model’s answer
# ----------------------------------------------------------------------
conv = [{"role": "system", "content":SYS_PROMPT}]
conv.append({"role": "user", "content": question})
print("\n--- Running reasoning model ---")
res = llm_stream(conv, thinking=True, retry_on_repeat=True, max_retries=2,
                 the_model=G_REASONING_MODEL, verbose=True)

model_reasoning = res["reasoning"]
model_answer = res["content"]  # final answer text

print("\n--- Model's answer ---")
print(model_answer)
print("\n\n\n")

# ----------------------------------------------------------------------
# Use the LLM judge to compare model answer to expected
# ----------------------------------------------------------------------
print("\n--- Judging answer ---")
judge_result = llm_judge(
    question=question,
    expected=expected,
    answer=model_answer,
    verbose=True,
)

show_judge(judge_result)

# ----------------------------------------------------------------------
# Optional: show reasoning snippet and token usage
# ----------------------------------------------------------------------
print("\n  [Model output stats]")
if res.get("reasoning"):
    print(f"  🧠 Reasoning: {res['reasoning'][:50]}..")
print(f"  ⏱ {res['time_taken']:.2f}s | 📊 {res['usage']['total_tokens']} tokens")
