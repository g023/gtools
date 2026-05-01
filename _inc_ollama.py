# Author: g023
# License: MIT

# creative ways to get a response from an ollama model that powers the underlying infra of https://github.com/g023/gtools/ as the interface to Ollama Server

import requests
import json
import os
import time
import random
import re
import base64
from typing import List, Dict, Optional, Generator, Union

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
G_LOOP_SIZE = 75
G_APPEND_PROMPT = ""      # e.g. "no_think" or "think:" for custom behaviour
G_HOST = "http://localhost:11434" # where is your ollama server?

# Two model profiles – one for fast streaming, one for reasoning
G_FAST_MODEL = "hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0"
G_REASONING_MODEL = "hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0"

# Default to fast model; switch to reasoning for thinking tests
G_MODEL = G_FAST_MODEL
G_THINKING = True        # set True for reasoning models

G_CONTEXT_WINDOW = 40000
G_MAX_OUTPUT_TOKENS = 16384
G_TEMP = 1.0
G_REQUEST_TIMEOUT = 600  # seconds for standard requests
G_STREAM_TIMEOUT = 600   # seconds for streaming requests

G_OPTIONS = {
    "num_predict": G_MAX_OUTPUT_TOKENS,
    "top_k": 95,
    "top_p": 0.95,
    "min_p": 0.35,
    "typical_p": 0.3,
    "repeat_last_n": 16384,
    "temperature": G_TEMP,
    "repeat_penalty": 15.2,
    "presence_penalty": 0.5,
    "frequency_penalty": 1.0,
    "mirostat": 2,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": True,
    "numa": False,
    "num_ctx": G_CONTEXT_WINDOW,
    "num_batch": 2,
    "low_vram": False,
    "vocab_only": False,
    "use_mmap": True,
    "use_mlock": True,
    "num_thread": 1,
}

_STREAM_DONE = object()
_ENABLE_EXP_LOGGING = False  # Set to True to log results to database in experiments

def set_experiment_logging(enabled: bool = True):
    """Enable or disable logging to database during experiments."""
    global _ENABLE_EXP_LOGGING
    _ENABLE_EXP_LOGGING = enabled

def log_experiment_result(session_id: Optional[int] = None, experiment_name: str = "", **metrics):
    """Log experiment metrics to database for analysis."""
    if not _ENABLE_EXP_LOGGING:
        return
    try:
        import db
        if session_id is None:
            session = db.get_active_session()
            session_id = session["id"] if session else None
        if session_id:
            insight = f"[{experiment_name}] " + " | ".join(f"{k}={v}" for k, v in metrics.items())
            db.add_session_insight(session_id, insight, confidence=0.85)
    except Exception as e:
        print(f"Warning: Could not log experiment result: {e}")

# ---------------------------------------------------------------------------
# IMAGE UTILITIES
# ---------------------------------------------------------------------------
def encode_image_to_base64(image_path: str, strict: bool = False) -> Optional[str]:
    try:
        if not os.path.exists(image_path):
            msg = f"Image file not found: {image_path}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"Warning: {msg}")
            return None
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        msg = f"Error encoding image '{image_path}': {e}"
        if strict:
            raise ValueError(msg) from e
        print(f"Warning: {msg}")
        return None

def create_image_message(text: str, image_paths: List[str], role: str = "user", strict: bool = False) -> Dict:
    images = [b64 for path in image_paths if (b64 := encode_image_to_base64(path, strict=strict))]
    if not images and image_paths and strict:
        raise ValueError(f"No valid images from paths: {image_paths}")
    return {"role": role, "content": text, "images": images}

# ---------------------------------------------------------------------------
# PROMPT SHUFFLING
# ---------------------------------------------------------------------------
def shuffle_sentences_outside_codeblocks(text: str) -> str:
    pattern = r'(```.*?```)'
    parts = re.split(pattern, text, flags=re.DOTALL)
    result_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            sentences = re.split(r'(?<=[.!?])\s+', part.strip())
            if len(sentences) > 1:
                random.shuffle(sentences)
                result_parts.append(' '.join(sentences))
            else:
                result_parts.append(part)
        else:
            result_parts.append(part)
    return ''.join(result_parts)

# ---------------------------------------------------------------------------
# IMPROVED LOOP DETECTION
# ---------------------------------------------------------------------------
def _detect_loop_in_text(text: str, threshold: int = 4, window_size: int = 300) -> bool:
    if len(text) < 100:
        return False

    # line‑level detection
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= threshold * 2:
        if len(set(lines[-threshold:])) == 1:
            return True
        for block_size in range(2, min(6, len(lines)//2 + 1)):
            last_block = lines[-block_size:]
            repeats = 0
            idx = len(lines) - block_size
            while idx >= 0 and lines[idx:idx+block_size] == last_block:
                repeats += 1
                idx -= block_size
            if repeats >= threshold:
                return True

    # substring‑diversity in recent window
    recent_text = text[-window_size:] if len(text) > window_size else text
    if len(recent_text) >= 100:
        unique_windows = set()
        for i in range(len(recent_text) - 20 + 1):
            unique_windows.add(recent_text[i:i+20])
            if len(unique_windows) > 8:
                break
        if len(unique_windows) <= 5:
            return True

    # sentence repetition (nearby duplicates)
    sentences = [s.strip() for s in re.split(r'[.!?]+', recent_text) if s.strip()]
    if len(sentences) >= 4:
        recent_sentences = sentences[-4:]
        if len(set(recent_sentences)) <= 2:
            return True

    return False

def _estimate_usage(reasoning: str, content: str, time_taken: float, ttft: Optional[float] = None) -> Dict:
    reasoning_tokens = round(len(reasoning) / 3.245) if reasoning else 0
    content_tokens = round(len(content) / 3.245) if content else 0
    total_tokens = reasoning_tokens + content_tokens
    speed = total_tokens / time_taken if time_taken > 0 else 0
    return {
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "total_tokens": total_tokens,
        "generation_speed": speed,
        "ttft_seconds": ttft,
    }

def get_ttft(result: Dict) -> Optional[float]:
    return result.get("usage", {}).get("ttft_seconds")

def _shuffle_conversation(conv: List[Dict]):
    for msg in conv:
        if msg.get("role") in ("user", "assistant"):
            msg["content"] = shuffle_sentences_outside_codeblocks(msg["content"])

# ---------------------------------------------------------------------------
# CORE OLLAMA (unchanged)
# ---------------------------------------------------------------------------
def ollama_get_models(host: str | None = None) -> List[str]:
    effective_host = _resolve_host(host)
    try:
        resp = requests.get(f"{effective_host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", []) if "name" in m]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models from Ollama: {e}")
        return []

def _resolve_host(host: str | None) -> str:
    return (host or os.getenv("OLLAMA_HOST") or G_HOST).rstrip('/')

def _parse_stream_line(raw_line: Union[bytes, str]) -> Optional[Union[Dict, object]]:
    if isinstance(raw_line, (bytes, memoryview, bytearray)):
        line: str = bytes(raw_line).decode('utf-8', errors='replace')
    else:
        line = str(raw_line)
    line = line.strip()
    if not line or line.startswith(':'):
        return None
    if line.startswith('data:'):
        payload = line[5:].strip()
        if not payload or payload == '[DONE]':
            return _STREAM_DONE
    else:
        payload = line
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"Failed to decode chunk: {raw_line!r}, error: {exc}")
        return None

def _is_reasoning_effort_unsupported_error(status_code: int, body: str) -> bool:
    if status_code != 400 or not isinstance(body, str):
        return False
    body_lower = body.lower()
    return any(phrase in body_lower for phrase in [
        "think value",
        "not supported",
        "does not support thinking",
        "does not support think",
        "does not support 'think'",
    ])

def chat_with_ollama(
    messages: List[Dict[str, str]],
    model: str = G_MODEL,
    host: str | None = None,
    stream: bool = False,
    reasoning_effort: str | None = None,
    options: Dict = G_OPTIONS,
    thinking: bool = G_THINKING,
    request_timeout: float | None = None,
    **kwargs
) -> Union[Dict, Generator[Dict, None, None]]:
    if G_APPEND_PROMPT and messages and messages[-1].get("role") == "user":
        messages[-1]["content"] += f"{G_APPEND_PROMPT}"

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": thinking,
        "options": options,
    }
    payload["options"]["num_ctx"] = payload["options"].get("num_ctx", G_CONTEXT_WINDOW)
    payload["options"]["num_predict"] = payload["options"].get("num_predict", G_MAX_OUTPUT_TOKENS)
    payload["options"]["temperature"] = payload["options"].get("temperature", G_TEMP)

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    payload.update(kwargs)

    headers = {"Content-Type": "application/json"}
    effective_host = _resolve_host(host)
    endpoint = f"{effective_host}/api/chat"

    print(f"Using model: {model}")
    print(f"Temperature: {payload['options']['temperature']}")
    print(f"Context window (num_ctx): {payload['options']['num_ctx']}")

    timeout = request_timeout or (G_STREAM_TIMEOUT if stream else G_REQUEST_TIMEOUT)

    try:
        if stream:
            return _stream_response(endpoint, headers, payload, timeout=timeout)
        else:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                body = getattr(getattr(e, "response", None), "text", "")
                if reasoning_effort and _is_reasoning_effort_unsupported_error(resp.status_code, body):
                    payload.pop("reasoning_effort", None)
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=600)
                    resp.raise_for_status()
                    return resp.json()
                raise
            return resp.json()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Could not connect to Ollama server at {effective_host}.") from e
    except requests.exceptions.Timeout as e:
        raise requests.exceptions.Timeout("Request to Ollama timed out.") from e
    except requests.exceptions.RequestException as e:
        raise e

def _stream_response(endpoint: str, headers: Dict, payload: Dict, timeout: float = 600) -> Generator[Dict, None, None]:
    payload["stream"] = True
    try:
        with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                body = getattr(getattr(e, "response", None), "text", "")
                print(f"Stream HTTP error ({resp.status_code}): {body[:200]}")
                raise
            for line in resp.iter_lines(decode_unicode=True):
                chunk = _parse_stream_line(line)
                if chunk is None:
                    continue
                if chunk is _STREAM_DONE:
                    break
                if not isinstance(chunk, dict):
                    continue
                yield chunk
                if chunk.get("done"):
                    break
                if (choices := chunk.get("choices")):
                    finish = choices[0].get("finish_reason")
                    if finish in {"stop", "length", "content_filter"}:
                        break
    except requests.exceptions.ChunkedEncodingError as e:
        print(f"Stream encoding error (connection may have closed): {e}")
        raise ConnectionError("Stream connection terminated unexpectedly") from e
    except requests.exceptions.ConnectionError as e:
        print(f"Stream connection error: {e}")
        raise

# ---------------------------------------------------------------------------
# NON‑STREAMING
# ---------------------------------------------------------------------------
def llm_nonstream(conv=None, thinking=True, options=G_OPTIONS, the_model=None):
    if conv is None:
        conv = []
    ret = {"reasoning": "", "content": "", "usage": {}, "time_taken": 0}
    effective_model = the_model or G_MODEL
    print("\n--- (Non-Streaming) ---")
    try:
        t0 = time.time()
        resp = chat_with_ollama(conv, model=effective_model, reasoning_effort="medium",
                                thinking=thinking, options=options)
        assert isinstance(resp, dict), "Expected non-stream response"
        msg = resp['message']
        ret["time_taken"] = time.time() - t0
        ret["reasoning"] = msg.get('thinking', '')
        ret["content"] = msg.get('content', '')
        if "</think>" in ret["content"]:
            reason_part, content_part = ret["content"].rsplit("</think>", 1)
            ret["reasoning"] = (ret["reasoning"] + reason_part).strip()
            ret["content"] = content_part.strip()
        ttft = resp.get("metrics", {}).get("ttft") if isinstance(resp.get("metrics"), dict) else None
        ret["usage"] = _estimate_usage(ret["reasoning"], ret["content"], ret["time_taken"], ttft=ttft)
    except Exception as e:
        print(f"Error: {e}")
    return ret

# ---------------------------------------------------------------------------
# INTELLIGENT STREAMING
# ---------------------------------------------------------------------------
def llm_stream(
    conv=None,
    thinking=True,
    options=G_OPTIONS,
    retry_on_repeat=False,
    the_model=G_MODEL,
    max_reasoning_tokens: int | None = None,
    reasoning_only: bool = False,
    max_stream_seconds: float = 120.0,
    verbose: bool = True,
    max_retries: int = 3,
) -> Dict:
    if conv is None:
        conv = []
    conv = [msg.copy() for msg in conv]
    options = options.copy()
    options.setdefault("num_ctx", G_CONTEXT_WINDOW)
    options.setdefault("temperature", G_TEMP)

    def _single_stream_attempt(conv_local, opts, think_enabled, deadline: Optional[float], start_time: float):
        reason = ""
        content = ""
        in_reasoning = True
        loop = False
        cap = False
        first_chunk_time = None
        try:
            stream = chat_with_ollama(
                conv_local, model=the_model, stream=True,
                thinking=think_enabled, options=opts)
        except Exception as e:
            print(f"Stream error: {e}")
            return "", "", True, False, None

        if verbose:
            print("Streaming: ", end="")
        for chunk in stream:
            if first_chunk_time is None:
                first_chunk_time = time.time()
            if deadline and time.time() > deadline:
                if verbose:
                    print("\n[Timeout reached]")
                break
            reasoning = ""
            content_piece = ""
            if 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {})
                reasoning = delta.get('reasoning', '')
                content_piece = delta.get('content', '')
            elif 'message' in chunk:
                msg = chunk['message']
                reasoning = msg.get('thinking', '')
                content_piece = msg.get('content', '')
            if reasoning:
                reason += reasoning
                if verbose:
                    print(reasoning, end="", flush=True)
                if len(reason) > 200 and _detect_loop_in_text(reason[-300:]):
                    loop = True
                    break
            if content_piece:
                if in_reasoning and content_piece:
                    if verbose:
                        print("\n--- End of Reasoning, Start of Content ---")
                    in_reasoning = False
                content += content_piece
                if verbose:
                    print(content_piece, end="", flush=True)
                if len(content) > 200 and _detect_loop_in_text(content[-300:]):
                    loop = True
                    break
            if (max_reasoning_tokens is not None
                    and in_reasoning
                    and (len(reason) / 3.245) >= max_reasoning_tokens):
                cap = True
                break
            if reasoning_only and "</think>" in content:
                break
        if verbose:
            print()
        ttft = (first_chunk_time - start_time) if first_chunk_time else None
        return reason, content, loop, cap, ttft

    reason = ""
    content = ""
    ttft_final = None
    t_start = time.time()
    deadline = t_start + max_stream_seconds if max_stream_seconds else None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            options["temperature"] = round(random.uniform(0.6, 1.5), 2)
            _shuffle_conversation(conv)
            if verbose:
                print(f"\n[Retry {attempt}/{max_retries}] Temp={options['temperature']}")

        reason, content, loop, cap, ttft = _single_stream_attempt(
            conv, options, thinking, deadline, t_start)
        if ttft_final is None:
            ttft_final = ttft

        if retry_on_repeat and loop and attempt < max_retries:
            continue

        if cap and max_reasoning_tokens is not None:
            if verbose:
                print("\n[Reasoning cap reached – requesting final answer]")
            assistant_msg = {
                "role": "assistant",
                "content": "",
                "thinking": reason,
            }
            conv.append(assistant_msg)
            followup_opts = options.copy()
            followup_opts["temperature"] = options.get("temperature", G_TEMP)
            try:
                stream2 = chat_with_ollama(
                    conv, model=the_model, stream=True,
                    thinking=False, options=followup_opts)
                content2 = ""
                if verbose:
                    print("Final answer: ", end="")
                for ch in stream2:
                    if deadline and time.time() > deadline:
                        break
                    piece = ""
                    if 'choices' in ch:
                        piece = ch['choices'][0].get('delta', {}).get('content', '')
                    elif 'message' in ch:
                        piece = ch['message'].get('content', '')
                    if piece:
                        content2 += piece
                        if verbose:
                            print(piece, end="", flush=True)
                content = content2
                if verbose:
                    print()
            except Exception:
                pass
            break

        if reasoning_only:
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[:idx]
            content = ""
            break

        break

    if "</think>" in content:
        reason_part, content_part = content.rsplit("</think>", 1)
        reason = (reason + reason_part).strip()
        content = content_part.strip()

    t = time.time() - t_start
    usage = _estimate_usage(reason, content, t, ttft=ttft_final)
    return {"reasoning": reason, "content": content, "usage": usage, "time_taken": t}

# ---------------------------------------------------------------------------
# REASONING‑ONLY (multi‑round) — improved
# ---------------------------------------------------------------------------
def llm_reasoning_only(
    conv=None,
    thinking=True,
    options=G_OPTIONS,
    the_model=G_MODEL,
    rounds: int = 1,
    verbose: bool = True,
    context_summary: str = "",
    **kwargs
) -> str | List[str]:
    """
    Multi-round reasoning-only extraction.

    Each round runs llm_stream with reasoning_only=True, captures the
    thinking/reasoning text, and feeds it back as context for the next round.

    If `context_summary` is provided (caveman-style compressed context),
    it is injected into the round>0 prompts instead of the generic
    "Continue reasoning" message, enabling token-efficient context carryover.

    Returns list of reasoning strings if rounds>1, else single string.
    """
    if conv is None:
        conv = []
    conv = [msg.copy() for msg in conv]
    all_reasons = []

    for r in range(rounds):
        if r > 0:
            if context_summary:
                # Use compressed context for token efficiency
                prompt = (
                    f"Prior context: {context_summary}\n"
                    f"Continue reasoning. Build on prior findings. "
                    f"Avoid repetition. Move toward definitive answer."
                )
            else:
                prompt = "Continue reasoning based on the above."
            conv.append({"role": "user", "content": prompt})

        # Extract max_reasoning_tokens from kwargs if provided, else use None (no cap)
        rt_kwargs = {k: v for k, v in kwargs.items() if k != "max_reasoning_tokens"}
        max_rt = kwargs.get("max_reasoning_tokens", None)
        res = llm_stream(
            conv=conv,
            thinking=thinking,
            options=options,
            reasoning_only=True,
            the_model=the_model,
            verbose=verbose,
            retry_on_repeat=False,
            max_reasoning_tokens=max_rt,
            **rt_kwargs,
        )
        round_reasoning = res["reasoning"]
        all_reasons.append(round_reasoning)

        # Detect loop across rounds — if reasoning barely changed, break
        if r > 0 and round_reasoning:
            prev = all_reasons[r - 1]
            # Simple overlap check: if >80% of words same, likely looping
            prev_words = set(prev.split())
            curr_words = set(round_reasoning.split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
                if overlap > 0.8:
                    if verbose:
                        print(f"\n[Round {r+1}: >80% word overlap with previous — stopping early]")
                    break

        conv.append({
            "role": "assistant",
            "content": "",
            "thinking": round_reasoning,
        })

    return all_reasons if rounds > 1 else all_reasons[0]

# ===========================================================================
# COMPREHENSIVE DEMO / TEST (runs only when script is executed directly)
# ===========================================================================
if __name__ == "__main__":
    available = ollama_get_models()
    print("Available models:", available)

    # Determine which models we can use for the demo
    fast_model = G_FAST_MODEL if G_FAST_MODEL in available else None
    reasoning_model = G_REASONING_MODEL if G_REASONING_MODEL in available else None

    # If the fast model isn't available, pick the first model from the list
    if fast_model is None and available:
        fast_model = available[0]
        print(f"Fast model not found, using {fast_model}")
    if reasoning_model is None and available:
        # Try to find any model with 'thinking' in the name
        for m in available:
            if 'thinking' in m.lower():
                reasoning_model = m
                break
        if reasoning_model is None:
            print("No reasoning model found; thinking-related tests will be skipped.")

    # -----------------------------------------------------------------------
    # 1. Basic streaming with loop detection and retry (use fast model)
    # -----------------------------------------------------------------------
    if fast_model:
        print("\n\n" + "="*60)
        print("1. SMART STREAM (loop detection ON, retry ON)")
        print("Prompt: 'List the numbers from 1 to 1000, one per line.'")
        print("(This often triggers a loop – the system will interrupt and retry.)")
        conv = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List the numbers from 1 to 1000, one per line."}
        ]
        res = llm_stream(
            conv,
            thinking=False,
            options=G_OPTIONS,
            retry_on_repeat=True,
            the_model=fast_model,
            max_retries=2,
            max_stream_seconds=30,
            verbose=True,
        )
        content_sample = res['content'][:200] + "..." if len(res['content']) > 200 else res['content']
        print(f"\n[RESULT] Content (first 200 chars): {content_sample}")
        print(f"Time: {res['time_taken']:.2f}s, Tokens: {res['usage']['total_tokens']}")

    # -----------------------------------------------------------------------
    # 2. Reasoning cap (needs a reasoning model)
    # -----------------------------------------------------------------------
    if reasoning_model:
        print("\n\n" + "="*60)
        print("2. REASONING CAP (max 30 tokens) – reasoning is cut off and final answer requested")
        conv = [
            {"role": "system", "content": "You are a poet."},
            {"role": "user", "content": "Explain why the sky is blue, then write a haiku about it."}
        ]
        res = llm_stream(
            conv,
            thinking=True,
            options=G_OPTIONS,
            the_model=reasoning_model,
            max_reasoning_tokens=30,
            verbose=True,
        )
        print(f"\n[RESULT] Reasoning snippet: {res['reasoning'][:200]}...")
        print(f"Content: {res['content']}")
    else:
        print("\n\n[SKIP] Reasoning cap test – no reasoning model available.")

    # -----------------------------------------------------------------------
    # 3. Multi‑round reasoning‑only (2 rounds)
    # -----------------------------------------------------------------------
    if reasoning_model:
        print("\n\n" + "="*60)
        print("3. MULTI‑ROUND REASONING‑ONLY (2 rounds)")
        conv = [
            {"role": "system", "content": "You are a mathematician."},
            {"role": "user", "content": "Prove that the square root of 2 is irrational."}
        ]
        multi = llm_reasoning_only(
            conv,
            thinking=True,
            options=G_OPTIONS,
            the_model=reasoning_model,
            rounds=2,
            verbose=True,
        )
        for i, r in enumerate(multi, 1):
            print(f"\n--- Round {i} reasoning (first 200 chars) ---")
            print(r[:200] + "...")
    else:
        print("\n\n[SKIP] Multi‑round reasoning test – no reasoning model available.")

    # -----------------------------------------------------------------------
    # 4. Pure reasoning‑only (1 round)
    # -----------------------------------------------------------------------
    if reasoning_model:
        print("\n\n" + "="*60)
        print("4. SINGLE‑ROUND REASONING‑ONLY")
        conv = [
            {"role": "system", "content": "You are a logical thinker."},
            {"role": "user", "content": "Should I use a linked list or an array for my project? Reason briefly."}
        ]
        reason = llm_reasoning_only(
            conv,
            thinking=True,
            options=G_OPTIONS,
            the_model=reasoning_model,
            rounds=1,
            verbose=True,
        )
        print(f"\n[RESULT] {reason[:300]}...")
    else:
        print("\n\n[SKIP] Single round reasoning test – no reasoning model available.")

    # -----------------------------------------------------------------------
    # 5. Non‑streaming fallback
    # -----------------------------------------------------------------------
    if fast_model:
        print("\n\n" + "="*60)
        print("5. NON‑STREAMING (use for quick fallback)")
        conv = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        res = llm_nonstream(conv, thinking=False, the_model=fast_model)
        print(f"[RESULT] {res['content']}")

    # -----------------------------------------------------------------------
    # 6. Image message creation (no actual streaming, just show the structure)
    # -----------------------------------------------------------------------
    print("\n\n" + "="*60)
    print("6. IMAGE MESSAGE CREATION (demo)")
    img_msg = create_image_message("Describe this picture:", ["/path/to/cat.jpg", "/path/to/dog.png"])
    print(f"Generated message: {img_msg}")
    # Add it to a conversation and pass to llm_stream with a multimodal model if desired.
