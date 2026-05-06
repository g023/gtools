#!/usr/bin/env python3
"""
Agentic CLI using bash as a tool, with user oversight for each command.
Enhanced with automatic verification and content review.
Uses only stock Python and the provided _inc_ollama, _json_fix, and _dag modules.
"""

import sys
import subprocess
import re
import os
from _inc_ollama import llm_stream
from _json_fix import fix_json_string
from _dag import process_dag_from_json, compute_topological_order


# ----------------------------------------------------------------------
# Utility: detect what files a command touches (deduplicated)
# ----------------------------------------------------------------------
def detect_affected_files(command):
    """
    Heuristic to find filenames created or modified by a command.
    Returns a dict: {filepath: 'created'|'appended'|'modified'}
    """
    files = {}
    
    # cat >> file (append via heredoc)
    for match in re.finditer(r'cat\s+>>\s*[\'\"]?([^\s;\'\"|&]+)[\'\"]?', command):
        path = match.group(1)
        if path and not path.startswith('/dev/'):
            files[path] = 'appended'
    
    # cat > file (overwrite via heredoc)
    for match in re.finditer(r'cat\s+>\s*[\'\"]?([^\s;\'\"|&]+)[\'\"]?', command):
        path = match.group(1)
        if path and not path.startswith('/dev/') and path not in files:
            files[path] = 'created'
    
    # tee file or tee -a file
    for match in re.finditer(r'tee\s+(-a\s+)?[\'\"]?([^\s;\'\"|&]+)[\'\"]?', command):
        path = match.group(2)
        if path and not path.startswith('/dev/') and path not in files:
            files[path] = 'appended' if match.group(1) else 'created'
    
    # Redirects: > file (overwrite) or >> file (append)
    for match in re.finditer(r'(?<!\<)\s*(>>?)\s*[\'\"]?([^\s;\'\"|&]+)[\'\"]?', command):
        op, path = match.group(1), match.group(2)
        if path and not path.startswith('/dev/') and path not in files:
            files[path] = 'appended' if op == '>>' else 'created'
    
    # touch file1 file2 ...
    touch_match = re.search(r'touch\s+(.+?)(?:\s*&&|\s*;|\s*\||$)', command)
    if touch_match:
        for f in re.findall(r'[\'\"]?([^\s;\'\"|&]+)[\'\"]?', touch_match.group(1)):
            if f and not f.startswith('/dev/') and f not in files:
                files[f] = 'modified'
    
    # mkdir dir
    for match in re.finditer(r'mkdir\s+(?:-p\s+)?[\'\"]?([^\s;\'\"|&]+)[\'\"]?', command):
        path = match.group(1)
        if path and not path.startswith('/dev/') and path not in files:
            files[path] = 'created'
    
    return files


def generate_verify_command(filepath, operation):
    """
    Generate appropriate verification for a file.
    Returns (verify_cmd, human_label) tuple.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.py':
        return (
            f"python3 -m py_compile {filepath} 2>&1 && echo 'SYNTAX_PASS' || echo 'SYNTAX_FAIL'",
            "Python syntax check"
        )
    elif ext == '.sh':
        return (
            f"bash -n {filepath} 2>&1 && echo 'SYNTAX_PASS' || echo 'SYNTAX_FAIL'",
            "Bash syntax check"
        )
    elif ext == '.json':
        return (
            f"python3 -c \"import json; json.load(open('{filepath}')); print('VALID_JSON')\" 2>&1 || echo 'INVALID_JSON'",
            "JSON validation"
        )
    elif ext in ('.js', '.mjs'):
        return (
            f"node --check {filepath} 2>&1 && echo 'SYNTAX_PASS' || echo 'SYNTAX_FAIL'",
            "JS syntax check"
        )
    else:
        # For text files, check and preview content
        return (
            f"test -f {filepath} && wc -l < {filepath} | xargs echo 'LINES:' && echo '---PREVIEW---' && head -5 {filepath}",
            "File content preview"
        )


# ----------------------------------------------------------------------
# DAG planning from the user goal
# ----------------------------------------------------------------------
def generate_dag(goal):
    """Ask the LLM to produce a DAG for the given goal."""
    system_prompt = (
        "You are an advanced system algorithm. "
        "Give answers with no fluff, and no introduction."
    )

    user_prompt = f"""You are given the following input statement:

=== input_statement ===
{goal}
=== /input_statement ===

Deconstruct this statement into a Directed Acyclic Graph (DAG).

## STEP 1: REASON ABOUT ORDERING
Before outputting JSON, reason through these questions:
1. What is the correct temporal or logical sequence of actions?
2. Are there any reversed dependencies that violate real-world logic?
3. Identify the true start node (no incoming edges) and end node (no outgoing edges).

## STEP 2: OUTPUT JSON
Return ONLY a valid JSON object with this exact structure:

{{
  "reasoning": "Brief explanation of the logical order",
  "dag": {{
    "nodes": [
      {{"id": "string", "label": "string", "type": "action|decision|start|end"}}
    ],
    "edges": [
      {{"from": "string", "to": "string", "condition": null, "reason": "string"}}
    ],
    "metadata": {{
      "is_acyclic": true,
      "cycle_explanation": "string",
      "parallel_paths": [],
      "topological_order": ["node_id1", "node_id2", "..."]
    }}
  }}
}}

CRITICAL RULES:
1. The start node is a marker only — put NO real work in it. First action node does actual work.
2. The end node MUST have NO outgoing edges and be LAST in topological_order.
3. No self-loops (from == to).
4. Only action nodes contain real work. start/end are structural.

Now process the input statement and return ONLY the JSON.
"""

    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    res = llm_stream(conv, thinking=True, max_reasoning_tokens=40, verbose=True)
    raw = res["content"]

    try:
        fixed = fix_json_string(raw)
    except Exception as e:
        print(f"⚠️  JSON fix failed ({e}), attempting raw parse …")
        fixed = raw

    dag = process_dag_from_json(fixed)

    if not dag.topological_order:
        order = compute_topological_order(dag.nodes, dag.edges)
    else:
        order = dag.topological_order

    return dag, order


# ----------------------------------------------------------------------
# Snapshot project files for context
# ----------------------------------------------------------------------
def get_file_inventory():
    """Return a summary of relevant files in the current directory."""
    lines = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden dirs and common non-project dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', 'venv')]
        for f in files:
            if f.startswith('.'):
                continue
            path = os.path.join(root, f)
            try:
                size = os.path.getsize(path)
                lines.append(f"  {path} ({size} bytes)")
            except:
                pass
    if not lines:
        return "(no files yet)"
    return "\n".join(lines[:20])  # Limit to 20 files


# ----------------------------------------------------------------------
# Verification loop
# ----------------------------------------------------------------------
def run_verification_loop(cmd, conv, goal, task_label):
    """
    After a command executes, detect affected files and verify them.
    For text files, show content preview so user can judge quality.
    """
    affected = detect_affected_files(cmd)
    
    if not affected:
        return True
    
    print(f"\n🔍 Verifying {len(affected)} file(s)...")
    
    for filepath, operation in affected.items():
        if not os.path.exists(filepath):
            print(f"  ⚠️  Expected file not found: {filepath}")
            continue
        
        verify_cmd, label = generate_verify_command(filepath, operation)
        print(f"  📄 {filepath} [{operation}] — {label}")
        
        try:
            proc = subprocess.run(
                verify_cmd, shell=True, capture_output=True, text=True,
                timeout=15, executable='/bin/bash'
            )
            output = (proc.stdout + proc.stderr).strip()
            
            print(f"  {'✅' if proc.returncode == 0 else '❌'} {label}:")
            # Print each line of output indented
            for line in output.split('\n')[:8]:
                print(f"     {line}")
            
            if proc.returncode != 0:
                # Feed failure back to agent
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    preview = content[:500]
                    if len(content) > 500:
                        preview += "\n... (truncated)"
                except:
                    preview = "(could not read file)"
                
                conv.append({
                    "role": "user",
                    "content": (
                        f"⚠️  Verification FAILED for {filepath}\n"
                        f"Error: {output[:300]}\n\n"
                        f"Current file content:\n{preview}\n\n"
                        f"Provide a CORRECTED command to fix this file."
                    )
                })
                
                # Retry loop
                for attempt in range(3):
                    print(f"\n  🔧 Asking agent for fix (attempt {attempt+1}/3)...")
                    res = llm_stream(conv, thinking=True, verbose=False)
                    fix_content = res["content"]
                    conv.append({"role": "assistant", "content": fix_content})
                    
                    fix_cmd = extract_command(fix_content)
                    if not fix_cmd:
                        print("  ⚠️  No command found in response")
                        break
                    
                    print(f"  💡 Agent proposes: {fix_cmd[:120]}...")
                    choice = input("  Apply fix? (y/n/e): ").strip().lower()
                    if choice == 'n':
                        break
                    elif choice == 'e':
                        fix_cmd = input("  Enter corrected command: ").strip()
                    
                    try:
                        subprocess.run(
                            fix_cmd, shell=True, capture_output=True, text=True,
                            timeout=30, executable='/bin/bash'
                        )
                        # Re-verify
                        vproc = subprocess.run(
                            verify_cmd, shell=True, capture_output=True, text=True,
                            timeout=15, executable='/bin/bash'
                        )
                        if vproc.returncode == 0:
                            print(f"  ✅ Fixed successfully")
                            break
                        print(f"  ❌ Still failing: {vproc.stderr[:200]}")
                        conv.append({
                            "role": "user",
                            "content": f"Fix attempt {attempt+1} failed. Error:\n{vproc.stderr[:300]}"
                        })
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        break
        
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Verification timed out")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    return True


def extract_command(content):
    """Extract a bash command from agent response."""
    # Multiline COMMAND ... END_COMMAND
    match = re.search(r"COMMAND:\s*\n(.*?)\nEND_COMMAND", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Single line variations
    match = re.search(r"COMMAND:\s*(.*?)\nEND_COMMAND", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"^COMMAND:\s*(.*?)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    return None


# ----------------------------------------------------------------------
# Agent loop for a single DAG node
# ----------------------------------------------------------------------
def execute_agent_for_task(goal, task_label, file_context=""):
    """
    Run interactive agent loop for one task.
    After each command, auto-verify affected files.
    """
    print(f"\n🤖 Working on: {task_label}")

    conv = [
        {
            "role": "system",
            "content": (
                "You are an autonomous agent with access to a bash shell. "
                "You will be given a task and must accomplish it by issuing NON-BLOCKING bash commands "
                "one at a time. For each step, output your reasoning, then provide the exact "
                "bash command to execute.\n\n"
                "CRITICAL FILE OPERATIONS:\n"
                "- Use 'cat > file << EOF' to CREATE/OVERWRITE a file\n"
                "- Use 'cat >> file << EOF' to APPEND to an existing file\n"
                "- Use 'cat file' to READ and check what's already there\n"
                "- Before modifying a file, READ it first to know its current contents\n\n"
                "IMPORTANT FOR MULTILINE COMMANDS:\n"
                "Use bash heredoc syntax:\n"
                "  cat > file.txt << 'EOF'\n"
                "  content here\n"
                "  EOF\n\n"
                "The closing delimiter MUST be on its own line with no spaces.\n\n"
                "Start with 'RATIONALE:' then 'COMMAND:' then the command, then 'END_COMMAND' on its own line.\n"
                "When task is complete, output 'DONE' on its own line.\n\n"
                "FORBIDDEN: nano, vim, and other blocking applications.\n"
                "Local Python is Python3. Avoid 'echo' for multiline content."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Overall goal: {goal}\n"
                f"Current subtask: {task_label}\n"
                f"{file_context}\n"
                "Start by assessing the situation. If files already exist, read them first.\n"
                "Use heredoc (<< 'EOF' ... EOF) for multiline content.\n"
                "End each command with 'END_COMMAND' on a new line."
            ),
        },
    ]

    done = False
    while not done:
        res = llm_stream(conv, thinking=True, verbose=False)
        content = res["content"]
        conv.append({"role": "assistant", "content": content})
        print("\n" + "─" * 50)
        
        # Show shorter preview of agent response
        preview = content[:400]
        if len(content) > 400:
            preview += f"\n... ({len(content)} chars total)"
        print("Agent:", preview)

        if re.search(r"\bDONE\b", content):
            print("✅ Subtask marked as completed.")
            done = True
            break

        cmd = extract_command(content)
        
        if not cmd:
            print("No command found in agent's response.")
            action = input("Continue? (y/n/e): ").strip().lower()
            if action == "n":
                done = True
                break
            elif action == "e":
                print("Enter command (Ctrl+D when done):")
                lines = []
                try:
                    while True:
                        lines.append(input())
                except EOFError:
                    pass
                cmd = "\n".join(lines).strip()
                if not cmd:
                    continue
            else:
                conv.append({
                    "role": "user",
                    "content": "Please provide the command with 'COMMAND:' followed by 'END_COMMAND'."
                })
                continue
        
        # Warn on overwrite vs append
        if re.search(r'cat\s+>\s+', cmd) and not re.search(r'>>', cmd):
            existing_file = re.search(r'cat\s+>\s*[\'\"]?([^\s;\'\"|&]+)[\'\"]?', cmd)
            if existing_file and os.path.exists(existing_file.group(1)):
                print(f"⚠️  WARNING: This will OVERWRITE existing file '{existing_file.group(1)}'")
                print("   Use 'cat >> file << EOF' to APPEND instead.")
        
        # Show command and get approval
        reasoning_match = re.search(r'RATIONALE:\s*(.*?)(?:COMMAND:|$)', content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "(No reasoning)"
        
        print(f"\n💡 {reasoning[:200]}")
        print(f"⚡ Command:\n{cmd[:300]}{'...' if len(cmd) > 300 else ''}")

        approved = False
        while True:
            choice = input("\nExecute? (y)es/(n)o/(e)dit/(r)ead-first/(s)kip/(q)uit: ").strip().lower()
            
            if choice in ("y", "yes"):
                approved = True
                break
            elif choice in ("n", "no"):
                conv.append({
                    "role": "user",
                    "content": "Command rejected. Suggest alternative."
                })
                break
            elif choice in ("e", "edit"):
                print("Enter new command (Ctrl+D when done):")
                lines = []
                try:
                    while True:
                        lines.append(input())
                except EOFError:
                    pass
                new_cmd = "\n".join(lines).strip()
                if new_cmd:
                    cmd = new_cmd
                    approved = True
                    break
            elif choice == "r":
                # Read the target file first
                target = re.search(r'(?:cat\s+[>>]?\s*|>+\s*)[\'\"]?([^\s;\'\"|&]+)[\'\"]?', cmd)
                if target and os.path.exists(target.group(1)):
                    with open(target.group(1), 'r') as f:
                        content = f.read()
                    print(f"\n📖 Current content of {target.group(1)}:")
                    print(content[:500])
                    print(f"... ({len(content)} chars total)" if len(content) > 500 else "")
                    conv.append({
                        "role": "user",
                        "content": f"Current content of {target.group(1)}:\n{content[:1000]}"
                    })
                else:
                    print("📖 File doesn't exist yet — safe to create")
                continue  # Back to approval
            elif choice in ("s", "skip"):
                conv.append({"role": "user", "content": "Skip this command."})
                done = True
                break
            elif choice in ("q", "quit"):
                sys.exit(0)
        
        if not approved and not done:
            continue
        if done:
            break

        # Execute
        print(f"\n🚀 Running...")
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30, executable='/bin/bash'
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, rc = "", "Command timed out", -1
        except Exception as e:
            stdout, stderr, rc = "", str(e), -1

        result_msg = f"Exit code: {rc}\n"
        if stdout:
            result_msg += f"STDOUT:\n{stdout[:500]}\n"
        if stderr:
            result_msg += f"STDERR:\n{stderr[:500]}\n"
        if not stdout and not stderr:
            result_msg += "(no output)"

        conv.append({"role": "user", "content": f"Command result:\n{result_msg}"})
        print("📋 " + result_msg[:200])

        # Auto-verify if command succeeded
        if rc == 0:
            run_verification_loop(cmd, conv, goal, task_label)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    else:
        goal = input("Enter your goal: ").strip()

    print("🧠 Generating plan …")
    try:
        dag, order = generate_dag(goal)
        
        # Filter out start and end nodes for execution
        actionable_order = [
            nid for nid in order 
            if dag.nodes[nid].type not in ('start', 'end')
        ]
        
        print("\n✅ Plan generated.")
        print(f"   All nodes: {' → '.join(order)}")
        if actionable_order:
            print(f"   Actionable: {' → '.join(actionable_order)}")
        else:
            print("   (all nodes are structural — nothing to execute)")
            return

        context = []

        for i, node_id in enumerate(actionable_order, 1):
            node = dag.nodes[node_id]
            print(f"\n{'='*60}")
            print(f"--- Step {i}/{len(actionable_order)}: {node.label} ---")
            print(f"{'='*60}")

            # Build context with file inventory
            file_context = "Current project files:\n" + get_file_inventory()
            if context:
                file_context += "\n\nPreviously completed:\n" + "\n".join(context)

            execute_agent_for_task(goal, node.label, file_context)

            summary = input("\nWhat was achieved? (Enter to skip): ").strip()
            if summary:
                context.append(f"✅ {node.label}: {summary}")
            else:
                context.append(f"✅ {node.label}: done.")

        print("\n🎉 Goal completed.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()