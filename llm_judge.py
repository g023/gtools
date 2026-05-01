#!/usr/bin/env python3
"""
llm_judge.py
Author: g023 - github.com/g023
License: MIT

LLM-as-judge evaluator for reasoning benchmarks.
Uses a language model to judge whether an answer matches the expected ground truth,
rather than relying on brittle keyword/pattern matching.
"""
import re
import json
from typing import Callable
from _inc_ollama import llm_stream, G_FAST_MODEL, G_REASONING_MODEL, ollama_get_models

# ----------------------------------------------------------------------
# Judge model selection
# ----------------------------------------------------------------------
_available = ollama_get_models()

# Prefer a capable model for judging; fall back to whatever is available
_JUDGE_MODEL_CANDIDATES = [
    G_REASONING_MODEL,
    G_FAST_MODEL,
]

JUDGE_MODEL: str | None = None
for candidate in _JUDGE_MODEL_CANDIDATES:
    if candidate and candidate in _available:
        JUDGE_MODEL = candidate
        break
if not JUDGE_MODEL and _available:
    JUDGE_MODEL = _available[0]

# ----------------------------------------------------------------------
# Judge prompt templates
# ----------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of LLM reasoning answers. Your job is to judge whether a model's answer correctly matches the expected ground truth for a given question.

Evaluate based on:
1. **Correctness** — Does the answer reach the right conclusion? (Most important)
2. **Reasoning quality** — Is the reasoning logical and complete?
3. **Precision** — Is the answer specific and accurate?

Be generous with partial credit — if the answer is essentially correct but worded differently, give full credit.
Be strict about factual errors — if the answer is wrong, mark it as wrong even if the reasoning sounds plausible.

Respond with a JSON object containing:
- "correct": true/false — whether the answer is correct
- "confidence": 0.0-1.0 — how confident you are in your judgment
- "reasoning": a brief explanation of your judgment
- "matches_expected": true/false — whether the answer matches the expected display value
"""

JUDGE_PROMPT_TEMPLATE = """Question: {question}

Expected correct answer / ground truth: {expected}

Model's answer: {answer}

Does the model's answer match the expected ground truth? Consider the answer correct if it reaches the same conclusion, even if worded differently. Be lenient with phrasing but strict with factual correctness.

Respond with a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "reasoning": "...", "matches_expected": true/false}}
"""

# ----------------------------------------------------------------------
# Structured judge result
# ----------------------------------------------------------------------
class JudgeResult:
    def __init__(self, correct: bool, confidence: float, reasoning: str,
                 matches_expected: bool, raw_response: str = ""):
        self.correct = correct
        self.confidence = confidence
        self.reasoning = reasoning
        self.matches_expected = matches_expected
        self.raw_response = raw_response

    def __bool__(self) -> bool:
        return self.correct

    def __repr__(self) -> str:
        status = "✅" if self.correct else "❌"
        return (f"JudgeResult({status} correct={self.correct}, "
                f"confidence={self.confidence:.2f}, "
                f"reasoning={self.reasoning[:60]}...)")

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "matches_expected": self.matches_expected,
        }


# ----------------------------------------------------------------------
# LLM judge function
# ----------------------------------------------------------------------
def llm_judge(
    question: str,
    expected: str,
    answer: str,
    model: str | None = None,
    verbose: bool = False,
    max_retries: int = 2,
) -> JudgeResult:
    """
    Use an LLM to judge whether `answer` matches the expected ground truth.

    Args:
        question: The original question asked.
        expected: The expected correct answer / ground truth.
        answer: The model's answer to evaluate.
        model: Which LLM to use as judge. Defaults to JUDGE_MODEL.
        verbose: Print debug info.
        max_retries: Number of retries on parse failure.

    Returns:
        JudgeResult with correctness judgment.
    """
    the_model = model or JUDGE_MODEL
    if not the_model:
        return JudgeResult(
            correct=False, confidence=0.0,
            reasoning="No judge model available",
            matches_expected=False,
        )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected=expected,
        answer=answer,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(max_retries + 1):
        try:
            resp = llm_stream(messages, thinking=False, the_model=the_model)
            text = resp["content"].strip()

            if verbose:
                print(f"\n[Judge raw response (attempt {attempt + 1})]:")
                print(text[:500])

            # Try to extract JSON from the response
            result = _parse_judge_response(text)

            if result is not None:
                if verbose:
                    print(f"[Judge parsed]: {result}")
                return result

            if verbose:
                print(f"[Judge] Failed to parse on attempt {attempt + 1}, retrying...")

        except Exception as e:
            if verbose:
                print(f"[Judge] Error on attempt {attempt + 1}: {e}")

    # Fallback: return a failed judgment
    return JudgeResult(
        correct=False, confidence=0.0,
        reasoning="Failed to get parseable judgment from LLM",
        matches_expected=False,
        raw_response=text if 'text' in locals() else "",
    )


def _parse_judge_response(text: str) -> JudgeResult | None:
    """Try to extract a JudgeResult from the LLM response text."""
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*"correct"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return JudgeResult(
                correct=bool(data.get("correct", False)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "")),
                matches_expected=bool(data.get("matches_expected", False)),
                raw_response=text,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try to find JSON with more flexible regex
    json_match2 = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match2:
        try:
            data = json.loads(json_match2.group())
            if "correct" in data:
                return JudgeResult(
                    correct=bool(data.get("correct", False)),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=str(data.get("reasoning", "")),
                    matches_expected=bool(data.get("matches_expected", False)),
                    raw_response=text,
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Heuristic: look for "correct: true" or "correct: false" patterns
    correct_match = re.search(r'"correct"\s*:\s*(true|false)', text, re.IGNORECASE)
    if correct_match:
        is_correct = correct_match.group(1).lower() == "true"
        # Try to extract confidence
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        # Try to extract reasoning
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        reasoning = reason_match.group(1) if reason_match else "Parsed from heuristic"
        return JudgeResult(
            correct=is_correct,
            confidence=confidence,
            reasoning=reasoning,
            matches_expected=is_correct,
            raw_response=text,
        )

    return None


# ----------------------------------------------------------------------
# Convenience: create an LLM-based check function for TEST_SUITE
# ----------------------------------------------------------------------
def make_llm_check(
    question: str,
    expected: str,
    model: str | None = None,
    confidence_threshold: float = 0.0,
    verbose: bool = False,
) -> Callable[[str], bool]:
    """
    Create a check function (Callable[[str], bool]) that uses an LLM judge
    to evaluate answers for a specific test case.

    This is designed to slot directly into the TEST_SUITE check field.

    Usage:
        {
            "name": "bat_and_ball",
            "question": "...",
            "check": make_llm_check(
                question="If a bat and a ball together cost $1.10...",
                expected="$0.05",
            ),
            "expected_display": "$0.05",
        }
    """
    def check(answer_text: str) -> bool:
        result = llm_judge(
            question=question,
            expected=expected,
            answer=answer_text,
            model=model,
            verbose=verbose,
        )
        return result.correct and result.confidence >= confidence_threshold
    return check


# ----------------------------------------------------------------------
# Batch judge: evaluate multiple answers at once
# ----------------------------------------------------------------------
def batch_judge(
    test_cases: list[dict],
    answers: list[str],
    model: str | None = None,
    verbose: bool = False,
) -> list[JudgeResult]:
    """
    Judge a batch of (question, expected, answer) triples.

    Args:
        test_cases: List of dicts with 'question' and 'expected_display' keys.
        answers: List of answer strings, same length as test_cases.
        model: Judge model to use.
        verbose: Print debug info.

    Returns:
        List of JudgeResult objects.
    """
    results = []
    for test, answer in zip(test_cases, answers):
        result = llm_judge(
            question=test["question"],
            expected=test["expected_display"],
            answer=answer,
            model=model,
            verbose=verbose,
        )
        results.append(result)
    return results


# ----------------------------------------------------------------------
# Demo / self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("🧑‍⚖️ LLM JUDGE MODULE")
    print(f"Judge model: {JUDGE_MODEL}")
    print()

    # Quick self-test
    test_question = "If a bat and a ball together cost $1.10, and the bat costs $1.00 more than the ball, how much does the ball cost?"
    test_expected = "$0.05"

    test_answers = [
        "The ball costs $0.05. The bat costs $1.05, so together they cost $1.10.",
        "The ball costs $0.10. The bat costs $1.00, so together they cost $1.10.",
        "I think the ball is 5 cents because if the bat is $1.05, that's $1 more than the ball.",
    ]

    print("Self-testing LLM judge:")
    for i, ans in enumerate(test_answers):
        result = llm_judge(test_question, test_expected, ans, verbose=True)
        print(f"\n  Answer {i + 1}: {ans[:60]}...")
        print(f"  Result: {result}")
        print(f"  Correct: {result.correct}, Confidence: {result.confidence:.2f}")
        print(f"  Reasoning: {result.reasoning}")
        print("-" * 60)
