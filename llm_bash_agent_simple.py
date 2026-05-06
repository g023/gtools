#!/usr/bin/env python3
"""
Agentic CLI using bash as a tool, with user oversight for each command.
The agent plans a DAG of actions and then executes them step by step.
Uses only stock Python and the provided _inc_ollama, _json_fix, and example_dag modules.

Author: g023 (https://github.com/g023)
License: MIT
"""

import sys
import subprocess
import re

from _inc_ollama import llm_stream
from _json_fix import fix_json_string
from _dag import process_dag_from_json, compute_topological_order


# ----------------------------------------------------------------------
# DAG planning from the user goal
# ----------------------------------------------------------------------
def generate_dag(goal):
    """
    Ask the LLM to produce a DAG for the given goal.
    Returns (ProcessedDAG, topological_order_list).
    """
    system_prompt = (
        "You are an advanced system algorithm. "
        "Give answers with no fluff, and no introduction."
    )

    user_prompt = f"""You are given the following input statement:

=== input_statement ===
{goal}
=== /input_statement ===

Deconstruct this statement into a Directed Acyclic Graph (DAG).

<ignore:no_think>

## STEP 1: REASON ABOUT ORDERING
Before outputting JSON, reason through these questions:
1. What is the correct temporal or logical sequence of actions?
2. Are there any reversed dependencies (e.g., "build before design") that violate real-world logic?
3. Identify the true start node (no incoming edges) and end node (no outgoing edges).

## STEP 2: OUTPUT JSON
Return ONLY a valid JSON object with the following exact structure:

{{
  "reasoning": "Brief explanation of the logical order and why edges are correct",
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

## CRITICAL RULES:
1. **End node rule**: A node with type="end" MUST have NO outgoing edges.
   The end node should be the LAST node in topological_order.
2. **Self-loop rule**: No edge where from == to.
3. **Single end node**: Use exactly ONE node with type="end".
   Do NOT create additional end nodes.
4. The final node in the sequence should have type="end" and appear only
   as a 'to' in edges, never as a 'from'.

Now process the input statement and return ONLY the JSON.
Make sure the end node appears only as a 'to' in edges, never as a 'from'.
"""

    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    res = llm_stream(conv, thinking=True, max_reasoning_tokens=40, verbose=True)
    raw = res["content"]

    # Fix common JSON issues before parsing
    try:
        fixed = fix_json_string(raw)
    except Exception as e:
        print(f"⚠️  JSON fix failed ({e}), attempting raw parse …")
        fixed = raw

    dag = process_dag_from_json(fixed)

    # Make sure we have a topological order
    if not dag.topological_order:
        order = compute_topological_order(dag.nodes, dag.edges)
    else:
        order = dag.topological_order

    return dag, order


# ----------------------------------------------------------------------
# Agent loop for a single DAG node (sub‑task) # removed heredoc cleaning
# ----------------------------------------------------------------------
def execute_agent_for_task(goal, task_label):
    """
    Run an interactive agent loop for one high‑level task.
    The agent proposes bash commands; the user must approve each one.
    """
    print(f"\n🤖 Working on: {task_label}")

    conv = [
        {
            "role": "system",
            "content": (
                "You are an autonomous agent with access to a bash shell. "
                "You will be given a task and must accomplish it by issuing NON-BLOCKING bash commands "
                "one at a time. For each step, output your reasoning, then provide the exact "
                "bash command to execute. "
                "\n\n"
                "IMPORTANT FOR MULTILINE COMMANDS: "
                "For commands spanning multiple lines, use bash heredoc syntax:\n"
                "  cat > file.txt << 'EOF'\n"
                "  line 1 content\n"
                "  line 2 content\n"
                "  line 3 content\n"
                "  EOF\n"
                "\n"
                "CRITICAL: The closing delimiter (EOF, END, etc.) MUST be on its own line "
                "with no leading or trailing spaces. Do not indent the closing delimiter.\n\n"
                "Start your response with 'RATIONALE:' followed by your reasoning. "
                "Then write 'COMMAND:' and on the following lines, write the full command. "
                "After the command, write 'END_COMMAND' on its own line. "
                "You can use multiple commands if needed, but only one per turn. "
                "After receiving the result of the command, you may ask for another. "
                "When the task is complete, output 'DONE' on its own line. "
                "If you need to view the current directory or files, use ls, pwd, etc. Local Python is Python3. "
                "Avoid using 'echo' - prefer printf, cat with heredoc, or direct file writing. "
                "FORBIDDEN: 'nano' and other blocking applications. "
                "Check before installing new things. "
                "Be safe and avoid destructive commands without confirmation."
                """
==desired_format==

RATIONALE:
Your reasoning here.

COMMAND:
cat > file.txt << 'EOF'
line 1 content
line 2 content
line 3 content
EOF

END_COMMAND

==/desired_format==
"""
            ),
        },
        {
            "role": "user",
            "content": (
                f"Overall goal: {goal}\n"
                f"Current subtask: {task_label}\n"
                "Start by assessing the situation and suggesting the first command.\n"
                "Remember: Use heredoc (<< 'EOF' ... EOF) for multiline content, avoid using 'echo'.\n"
                "IMPORTANT: Always end your command with 'END_COMMAND' on a new line."
            ),
        },
    ]

    done = False
    while not done:
        # Get the agent's next message
        res = llm_stream(conv, thinking=True, verbose=False)
        content = res["content"]
        conv.append({"role": "assistant", "content": content})
        print("Agent:", content)

        # Check for completion marker
        if re.search(r"\bDONE\b", content):
            print("✅ Subtask marked as completed.")
            done = True
            break

        # Try different patterns to extract the command
        cmd = None
        
        # Pattern 1: COMMAND: ... END_COMMAND (multiline)
        pattern1 = r"COMMAND:\s*\n(.*?)\nEND_COMMAND"
        match1 = re.search(pattern1, content, re.DOTALL | re.MULTILINE)
        if match1:
            cmd = match1.group(1).strip()
        
        # Pattern 2: COMMAND: on one line, then command, then END_COMMAND
        if not cmd:
            pattern2 = r"COMMAND:\s*(.*?)\nEND_COMMAND"
            match2 = re.search(pattern2, content, re.DOTALL | re.MULTILINE)
            if match2:
                cmd = match2.group(1).strip()
        
        # Pattern 3: Just COMMAND: followed by command (no END_COMMAND)
        if not cmd:
            pattern3 = r"COMMAND:\s*\n(.*?)(?=\n\n|$)"
            match3 = re.search(pattern3, content, re.DOTALL | re.MULTILINE)
            if match3:
                cmd = match3.group(1).strip()
                print("⚠️  Note: Command missing 'END_COMMAND' marker")
        
        # Pattern 4: Single line command
        if not cmd:
            single_line_match = re.search(r"^COMMAND:\s*(.*?)$", content, re.MULTILINE)
            if single_line_match:
                cmd = single_line_match.group(1).strip()
        
        if not cmd:
            # No command found, ask user
            print("No command found in agent's response.")
            action = input("Continue? (y/n/e to edit response): ").strip().lower()
            if action == "n":
                done = True
                break
            elif action == "e":
                print("Enter the correct command (multiple lines allowed, press Ctrl+D or Ctrl+Z then Enter when done):")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                cmd = "\n".join(lines).strip()
                if not cmd:
                    continue
            else:
                conv.append({
                    "role": "user",
                    "content": "Please provide the command with 'COMMAND:' followed by the command and then 'END_COMMAND' on a new line."
                })
                continue
        
        # Check for heredoc syntax issues
        if "<<" in cmd and "EOF" in cmd:
            # Validate heredoc syntax
            lines = cmd.split('\n')
            heredoc_started = False
            for i, line in enumerate(lines):
                if '<<' in line and ("'EOF'" in line or '"EOF"' in line or 'EOF' in line):
                    heredoc_started = True
                    # Check if the closing delimiter exists later
                    has_closing = False
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() == 'EOF' or lines[j].strip() == "EOF" or lines[j].strip() == "'EOF'":
                            has_closing = True
                            break
                    if not has_closing:
                        print("⚠️  Warning: Heredoc started but missing closing 'EOF'")
                        if input("Add closing 'EOF'? (y/n): ").lower() == 'y':
                            cmd += "\nEOF"
        
        # Check for echo command (discourage usage)
        if re.search(r'\becho\s+', cmd, re.MULTILINE):
            print("⚠️  Warning: 'echo' detected. Consider using heredoc (<< 'EOF') or printf for multiline content.")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                conv.append({
                    "role": "user",
                    "content": "Please rewrite your command without using 'echo'. Use heredoc syntax (<< 'EOF' ... EOF) or printf instead."
                })
                continue
        
        # Extract reasoning (everything before COMMAND:)
        reasoning = content.split("COMMAND:")[0].strip()
        # Remove RATIONALE: prefix if present
        reasoning = re.sub(r'^RATIONALE:\s*', '', reasoning, flags=re.MULTILINE)
        if not reasoning:
            reasoning = "(No reasoning provided)"

        # Show the user what will be executed
        print(f"\n💡 Reasoning: {reasoning}")
        print(f"⚡ Proposed command:\n{cmd}")

        approved = False
        while True:
            choice = (
                input("\nExecute? (y)es / (n)o / (e)dit / (s)kip / (q)uit: ")
                .strip()
                .lower()
            )

            if choice in ("y", "yes"):
                approved = True
                break

            elif choice in ("n", "no"):
                conv.append(
                    {
                        "role": "user",
                        "content": "Command rejected. Please suggest an alternative command. Consider using heredoc for multiline content and avoid 'echo'."
                    }
                )
                break

            elif choice in ("e", "edit"):
                print("Enter the new command (multiple lines allowed, press Ctrl+D or Ctrl+Z then Enter when done):")
                print("Tip: Use heredoc syntax for multiline content: << 'EOF' ... EOF")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                new_cmd = "\n".join(lines).strip()
                if new_cmd:
                    cmd = new_cmd
                    approved = True
                    break
                else:
                    print("Empty command. Try again.")

            elif choice in ("s", "skip"):
                conv.append(
                    {
                        "role": "user",
                        "content": "Skip this command and consider the subtask completed."
                    }
                )
                done = True
                break

            elif choice in ("q", "quit"):
                sys.exit(0)

            else:
                print("Invalid choice.")

        if not approved and not done:
            continue          # agent will propose a new command
        if done:
            break

        # Execute the approved command as-is
        print(f"Running command...")
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30, executable='/bin/bash'
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, rc = "", "Command timed out", -1
        except Exception as e:
            stdout, stderr, rc = "", str(e), -1

        # Build a result message for the agent
        result_msg = f"Exit code: {rc}\n"
        if stdout:
            result_msg += f"STDOUT:\n{stdout}\n"
        if stderr:
            result_msg += f"STDERR:\n{stderr}\n"
        if not stdout and not stderr:
            result_msg += "(no output)"

        conv.append({"role": "user", "content": f"Command result:\n{result_msg}"})
        print("📋 Output:\n" + result_msg)

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
        print("\n✅ Plan generated. Topological order:", " → ".join(order))

        # === NEW: execute every node, regardless of type ===
        context = []  # collect summaries of completed subtasks

        for i, node_id in enumerate(order, 1):
            node = dag.nodes[node_id]
            print(f"\n--- Step {i}/{len(order)}: {node.label} ---")

            # Pass previous context to the agent so it builds on earlier work
            task_description = node.label
            if context:
                task_description += (
                    "\n\nPrevious steps completed:\n" + "\n".join(context)
                )

            execute_agent_for_task(goal, task_description)

            # After the subtask finishes, ask the user for a one‑line summary
            summary = input(
                "What was achieved? (one sentence, or press Enter to skip): "
            ).strip()
            if summary:
                context.append(f"{node.label}: {summary}")
            else:
                context.append(f"{node.label}: completed.")

        print("\n🎉 Goal completed.")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()