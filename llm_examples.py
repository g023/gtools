# llm_examples.py
# Author: g023 (github.com/g023)
# License: MIT
# 20 powerful streaming examples using the functions from _inc_ollama.py

from _inc_ollama import (
    llm_stream,
    llm_nonstream,
    llm_reasoning_only,
    ollama_get_models,
    create_image_message,
    shuffle_sentences_outside_codeblocks,
    G_OPTIONS,
    G_FAST_MODEL,
    G_REASONING_MODEL,
    set_experiment_logging,
    log_experiment_result,
)
import time, os, sys

set_experiment_logging(True)   # optional, requires db module

# ---------------------- Model selection ----------------------
available = ollama_get_models()
fast_model = G_FAST_MODEL if G_FAST_MODEL in available else (available[0] if available else None)
reasoning_model = G_REASONING_MODEL if G_REASONING_MODEL in available else None
if not reasoning_model:
    for m in available:
        if 'thinking' in m.lower():
            reasoning_model = m
            break

if not fast_model:
    print("No usable model found. Exiting.")
    sys.exit(1)

print(f"Fast model: {fast_model}")
print(f"Reasoning model: {reasoning_model or '(not available — some examples will be skipped)'}")

# ---------------------- Helper for clean output ----------------------
def show(res, label):
    print(f"\n  [{label}]")
    if res.get("reasoning"):
        print(f"  🧠 Reasoning snippet: {res['reasoning'][:100]}...")
    print(f"  💬 Content snippet: {res['content'][:150]}...")
    print(f"  ⏱ {res['time_taken']:.2f}s | 📊 {res['usage']['total_tokens']} tokens")

# =====================================================================
# 1. Smart streaming with loop detection & retry (automatic prompt shuffle)
# =====================================================================
print("\n\n===== 1. Loop detection & retry (streaming visible) =====")
conv = [{"role":"user","content":"List the numbers 1 to 10, one per line. Then repeat that entire list three more times."}]
res = llm_stream(conv, thinking=False, options=G_OPTIONS, retry_on_repeat=True,
                 max_retries=2, max_stream_seconds=30, the_model=fast_model, verbose=True)
show(res, "Loop retry")
log_experiment_result(experiment_name="loop_retry", tokens=res["usage"]["total_tokens"],
                      time=res["time_taken"])

# =====================================================================
# 2. Reasoning cap - cut thinking after 30 tokens, finish answer
# =====================================================================
print("\n\n===== 2. Reasoning cap (30 tokens) - streamed =====")
if reasoning_model:
    conv = [{"role":"system","content":"You are a poet."},
            {"role":"user","content":"Explain why the sky is blue, then write a haiku about it."}]
    res = llm_stream(conv, thinking=True, options=G_OPTIONS, max_reasoning_tokens=30,
                     the_model=reasoning_model, verbose=True)
    show(res, "Haiku with capped reasoning")
else:
    print("Skipped - no reasoning model.")

# =====================================================================
# 3. Multi-round reasoning-only (3 rounds) - streamed reasoning
# =====================================================================
print("\n\n===== 3. Multi-round reasoning-only (3 rounds) =====")
if reasoning_model:
    conv = [{"role":"system","content":"You are a mathematician."},
            {"role":"user","content":"Prove √2 is irrational."}]
    thoughts = llm_reasoning_only(conv, thinking=True, options=G_OPTIONS,
                                  the_model=reasoning_model, rounds=3, verbose=True)
    for i, t in enumerate(thoughts, 1):
        print(f"  Round {i} final reasoning (first 100 chars): {t[:100]}...")
else:
    print("Skipped - no reasoning model.")

# =====================================================================
# 4. Chain: brainstorm (capped reasoning) → summarise (streamed)
# =====================================================================
print("\n\n===== 4. Brainstorm → Summarise chain =====")
if reasoning_model:
    conv = [{"role":"user","content":"Brainstorm 5 startup ideas for AI in healthcare."}]
    step1 = llm_stream(conv, thinking=True, max_reasoning_tokens=40,
                       the_model=reasoning_model, verbose=True)
    conv.append({"role":"assistant","content": step1["content"]})
    conv.append({"role":"user","content":"Summarise each idea in one sentence."})
    step2 = llm_stream(conv, thinking=False, the_model=reasoning_model, verbose=True)
    show(step2, "Summary of ideas")

# =====================================================================
# 5. Image message creation (demo structure, no streaming)
# =====================================================================
print("\n\n===== 5. Image message structure (no streaming) =====")
img_msg = create_image_message("What is in this image?", ["photo.jpg", "scan.png"])
print("Message with images:", img_msg)

# =====================================================================
# 6. Temperature sweep - compare low / high creativity (streamed)
# =====================================================================
print("\n\n===== 6. Temperature sweep - streamed =====")
poem_conv = [{"role":"user","content":"Write a short poem about the moon."}]
opts_low = {**G_OPTIONS, "temperature": 0.2}
opts_high = {**G_OPTIONS, "temperature": 1.5}
print("--- Low temperature (0.2) ---")
res_low = llm_stream(poem_conv, thinking=False, options=opts_low, the_model=fast_model, verbose=True)
print("--- High temperature (1.5) ---")
res_high = llm_stream(poem_conv, thinking=False, options=opts_high, the_model=fast_model, verbose=True)
print("\nLow temp:", res_low["content"][:150])
print("High temp:", res_high["content"][:150])

# =====================================================================
# 7. Non-streaming quick Q&A (not streamed, but fast)
# =====================================================================
print("\n\n===== 7. Non-streaming quick Q&A =====")
conv = [{"role":"user","content":"Capital of France?"}]
res = llm_nonstream(conv, thinking=False, the_model=fast_model)
show(res, "Fast non-stream")

# =====================================================================
# 8. Reasoning-only → feed reasoning into a new prompt for final answer
# =====================================================================
print("\n\n===== 8. Reasoning feedback loop =====")
if reasoning_model:
    base_conv = [{"role":"user","content":"Should I use a microservice or monolith?"}]
    reason = llm_reasoning_only(base_conv, thinking=True, the_model=reasoning_model,
                                rounds=1, verbose=True)
    follow = [{"role":"user","content": f"Based on this reasoning:\n{reason}\n\nGive a final recommendation."}]
    decision = llm_stream(follow, thinking=False, the_model=fast_model, verbose=True)
    show(decision, "Decision after reasoning")

# =====================================================================
# 9. Loop detection inside reasoning (streamed)
# =====================================================================
print("\n\n===== 9. Loop detection inside reasoning =====")
if reasoning_model:
    conv = [{"role":"user","content":"Repeat 50 times: 'I think therefore I am.'"}]
    res = llm_stream(conv, thinking=True, retry_on_repeat=True, max_retries=2,
                     the_model=reasoning_model, verbose=True)
    show(res, "Repetition loop broken")
else:
    print("Skipped - no reasoning model.")

# =====================================================================
# 10. Large generation stopped by timeout (streamed)
# =====================================================================
print("\n\n===== 10. Large generation cut by timeout =====")
conv = [{"role":"user","content":"Write a 10000-word essay on the history of bread."}]
res = llm_stream(conv, thinking=False, max_stream_seconds=8.0, the_model=fast_model, verbose=True)
print(f"\nStopped after {len(res['content'])} characters (timeout active).")

# =====================================================================
# 11. Experiment logging (no streaming)
# =====================================================================
print("\n\n===== 11. Experiment logging =====")
log_experiment_result(experiment_name="demo_run", model=fast_model, temperature=1.0)
print("Logged a demo metric.")

# =====================================================================
# 12. Multi-turn conversation with thinking enabled (streamed)
# =====================================================================
print("\n\n===== 12. Multi-turn with thinking =====")
if reasoning_model:
    conv = [{"role":"system","content":"You are a wise philosopher."},
            {"role":"user","content":"What is the meaning of life?"}]
    ans1 = llm_stream(conv, thinking=True, max_reasoning_tokens=50,
                      the_model=reasoning_model, verbose=True)
    conv.append({"role":"assistant","content": ans1["content"]})
    conv.append({"role":"user","content":"Now relate that to modern science."})
    ans2 = llm_stream(conv, thinking=False, the_model=reasoning_model, verbose=True)
    show(ans2, "Science connection")

# =====================================================================
# 13. Compare fast model vs reasoning model on same prompt
# =====================================================================
print("\n\n===== 13. Model comparison (streamed) =====")
q = [{"role":"user","content":"Explain gravity in 2 sentences."}]
print("  --- Fast model ---")
res_fast = llm_stream(q, thinking=False, the_model=fast_model, verbose=True)
if reasoning_model:
    print("  --- Reasoning model ---")
    res_reason = llm_stream(q, thinking=True, max_reasoning_tokens=30,
                            the_model=reasoning_model, verbose=True)
    print(f"\nFast: {res_fast['content'][:200]}")
    print(f"Reasoning: {res_reason['content'][:200]}")
else:
    print("Reasoning model not available for comparison.")

# =====================================================================
# 14. Manual prompt shuffle (no streaming)
# =====================================================================
print("\n\n===== 14. Manual prompt shuffle =====")
original = "Hello. I want to list numbers. 1, 2, 3, 4, 5, 6, 7, 8, 9."
shuffled = shuffle_sentences_outside_codeblocks(original)
print("Original:", original)
print("Shuffled:", shuffled)

# =====================================================================
# 15. Plan with multi-round reasoning, then execute with non-thinking
# =====================================================================
print("\n\n===== 15. Plan → Execute (streamed) =====")
if reasoning_model:
    plan_conv = [{"role":"user","content":"Plan a healthy weekly dinner menu."}]
    print("  --- Planning (reasoning) ---")
    plan = llm_reasoning_only(plan_conv, thinking=True, the_model=reasoning_model,
                              rounds=2, verbose=True)
    exec_conv = [{"role":"user","content": f"Using this plan:\n{plan[0]}\n\nWrite a shopping list."}]
    print("  --- Generating shopping list ---")
    shop_list = llm_stream(exec_conv, thinking=False, the_model=fast_model, verbose=True)
    show(shop_list, "Shopping list from plan")
else:
    print("Skipped - no reasoning model.")

# =====================================================================
# 16. Offline loop detection test (no streaming)
# =====================================================================
print("\n\n===== 16. Offline loop detection =====")
from _inc_ollama import _detect_loop_in_text
loop_text = "ZzZzZzZzZzZzZzZzZzZzZzZz" * 20
print(f"Loop detected in monotone string? {_detect_loop_in_text(loop_text)}")

# =====================================================================
# 17. Real image message (if a test image exists)
# =====================================================================
print("\n\n===== 17. Real image message (if file present) =====")
if os.path.exists("test.jpg"):
    msg = create_image_message("Describe this image.", ["test.jpg"])
    conv = [msg]
    print("  --- Streaming image description (model must support images) ---")
    res = llm_stream(conv, thinking=False, the_model=fast_model, verbose=True)
    show(res, "Image description")
else:
    print("No 'test.jpg' found - skipping image example.")

# =====================================================================
# 18. Conditional experiment log after reasoning cap
# =====================================================================
print("\n\n===== 18. Conditional log when reasoning cap is hit =====")
if reasoning_model:
    conv = [{"role":"user","content":"Explain quantum entanglement."}]
    res = llm_stream(conv, thinking=True, max_reasoning_tokens=20,
                     the_model=reasoning_model, verbose=True)
    if len(res["reasoning"]) > 60:
        log_experiment_result(experiment_name="cap_detected", reasoning_len=len(res["reasoning"]))
        print("Reasoning cap triggered!")
    show(res, "After cap")
else:
    print("Skipped.")

# =====================================================================
# 19. Multi-round reasoning → final one-paragraph summary
# =====================================================================
print("\n\n===== 19. Multi-round reasoning → summary =====")
if reasoning_model:
    conv = [{"role":"user","content":"Explain the theory of relativity."}]
    print("  --- Reasoning rounds ---")
    reasons = llm_reasoning_only(conv, thinking=True, the_model=reasoning_model,
                                 rounds=2, verbose=True)
    summary_conv = [{"role":"user","content": f"Reasoning:\n{reasons[0]}\n\nNow give a one-paragraph summary."}]
    print("  --- Summary ---")
    final = llm_stream(summary_conv, thinking=False, the_model=fast_model, verbose=True)
    show(final, "Relativity summary")

# =====================================================================
# 20. Full experiment: loop detection with TTFB measurement
# =====================================================================
print("\n\n===== 20. TTFB & loop experiment =====")
conv = [{"role":"user","content":"Count from 1 to 50."}]
t0 = time.time()
res = llm_stream(conv, thinking=False, retry_on_repeat=True, max_retries=3,
                 the_model=fast_model, max_stream_seconds=20, verbose=True)
ttfb = (time.time() - t0) - res["time_taken"]
print(f"\n  Rough TTFB: {ttfb:.3f}s")
show(res, "Counting test")
log_experiment_result(experiment_name="counting_loop", tokens=res["usage"]["total_tokens"],
                      ttfb=ttfb, time=res["time_taken"])

print("\nAll 20 examples completed.")
