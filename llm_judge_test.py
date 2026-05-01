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
A farmer wants to cross a river and take with him a wolf, a goat and a cabbage. 
He has a boat with three secure separate compartments. 
If the wolf and the goat are alone on one shore, the wolf will eat the goat. 
If the goat and the cabbage are alone on the shore, the goat will eat the cabbage. 
How can the farmer efficiently bring the wolf, the goat and the cabbage across the river without anything being eaten?
[[/question]]

[[right_answer]]
This is not the classic problem, and is a derivative of a much simpler solution:
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

SYS_PROMPT = """You are a clever expert and concise reasoner and a master at playing the devils' advocate on any concept.
You identify the single key insight that makes a problem simple.
You carefully check definitions of every term in a question.
You think out complex problems step-by-step. 
You identify and avoid common reasoning traps. 
You verify, validate, reflect, and course correct. 
What is the gotcha? What does a second look or a reflection tell us? 
You argue against common misconceptions, and dive into the overall picture. 
You will give your answer after short structured thinking of the problem.
You are not repetitive and will approach the problem from many angles and using different forms of structured thought in creative and imaginative ways to reframe the problem. 
""" + CAVEMAN_SYS

# ---------------------------------------------
# -------------------------
# Get the model’s answer
# ----------------------------------------------------------------------
# conv = [{"role": "system", "content":SYS_PROMPT}]


prompt = f"""Return just the list and nothing else. Start with the phrase "- " and then finish the list. 
Consider the order of information as being more important in priority in which it is presented. 
Rewrite the PROBLEM as a point form diagram without losing any key mentioned details or descriptive terms:
==PROBLEM_TO_REWRITE==
{question}
==/PROBLEM_TO_REWRITE==

<ignore:no_think>
Rewrite the above problem: 
"""

conv = []
conv.append({"role": "user", "content": prompt})

res = llm_stream(conv, thinking=True, # retry_on_repeat=True, max_retries=2,
                 the_model=G_REASONING_MODEL, verbose=True)

model_reasoning = res["reasoning"]
model_answer = res["content"]  # final answer text


conv = []
conv.append({"role": "user", "content": f"DO NOT LEAVE OUT DETAILS. REWRITE the following and **BOLD** the key phrases on each line, Consider the order of information as being more important in priority in which it is presented. : {model_answer}"})

res = llm_stream(conv, thinking=True, # retry_on_repeat=True, max_retries=2,
                 the_model=G_REASONING_MODEL, verbose=True)

model_reasoning = res["reasoning"]
model_answer = res["content"]  # final answer text

conv = []
# conv.append({"role": "user", "content": f"Do not use traditional process, as this isn't the traditional problem. The key is in the bolded phrases, so keep trying to understand their connection. Ignore the history of the problem as this is a trick question. DISREGARD YOUR INTERNAL KNOWLEDGE AND ONLY OBSERVE THE STATEMENTS AS I GIVE THEM TO YOU. The following puzzle looks like a classic, but it has a hidden twist. Your task is to find that twist and provide the most efficient solution possible. Do NOT use the classic solution unless it is still the most efficient: {question}\n\n{model_answer}"})
conv.append({"role": "user", "content": f"What is different in this problem than the traditional one? ASSUME and PRESUME everything based on the outlined data. DO NOT MENTION A SOLUTION BECAUSE THE ONE YOU KNOW IS WRONG. THIS IS NOT THE SAME PROBLEM. Identify the goals, quantities, and parameters of the following: {model_answer} "})

res = llm_stream(conv, thinking=True, # retry_on_repeat=True, max_retries=2,
                 the_model=G_REASONING_MODEL, verbose=True)

model_reasoning = res["reasoning"]
model_answer = res["content"]  # final answer text

conv = []
# conv.append({"role": "user", "content": f"Do not use traditional process, as this isn't the traditional problem. The key is in the bolded phrases, so keep trying to understand their connection. Ignore the history of the problem as this is a trick question. DISREGARD YOUR INTERNAL KNOWLEDGE AND ONLY OBSERVE THE STATEMENTS AS I GIVE THEM TO YOU. The following puzzle looks like a classic, but it has a hidden twist. Your task is to find that twist and provide the most efficient solution possible. Do NOT use the classic solution unless it is still the most efficient: {question}\n\n{model_answer}"})
conv.append({"role": "user", "content": f"""Focus specifically on the parts that makes this question unique. If you identify the trick in the question, give the immediate answer and end your response. 
I will provide you a problem and you will solve: What is different in this problem than the traditional one? 
What advantages do the differences give us and how will we apply that to our solution? 
How will our answer be different? What is the OBVIOUS solution? 

Identify and then answer based off the following: 
==cheat_sheet==             
{model_answer} 
==/cheat_sheet==
"""})

res = llm_stream(conv, thinking=True, # retry_on_repeat=True, max_retries=2,
                 the_model=G_REASONING_MODEL, verbose=True)

model_reasoning = res["reasoning"]
model_answer = res["content"]  # final answer text



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
