#!/usr/bin/env python3
"""
Agentic CLI using bash as a tool, with user oversight for each command.
The agent plans a DAG of actions and then executes them step by step.
Uses only stock Python and the provided _inc_ollama, _json_fix, and example_dag modules.

Enhanced with memory management and reasoning/actionable step classification.

Author: g023 (https://github.com/g023)
License: MIT
"""

import sys
import subprocess
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from _inc_ollama import llm_stream
from _json_fix import fix_json_string
from _dag import process_dag_from_json, compute_topological_order, NodeCategory, ProcessedDAG, DAGNode

@dataclass
class StepMemory:
    """Local memory for a single step execution"""
    step_id: str
    step_label: str
    category: NodeCategory
    summary: str = ""
    artifacts: List[str] = field(default_factory=list)  # Files created/modified
    decisions: Dict[str, str] = field(default_factory=dict)  # Key decisions made
    observations: List[str] = field(default_factory=list)  # Important observations
    exit_status: Optional[int] = None
    
    def to_context_string(self) -> str:
        """Convert memory to a compact context string for the LLM"""
        parts = [f"Step: {self.step_label}"]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.artifacts:
            parts.append(f"Created/Modified: {', '.join(self.artifacts)}")
        if self.decisions:
            decisions_str = ', '.join([f"{k}={v}" for k, v in self.decisions.items()])
            parts.append(f"Decisions: {decisions_str}")
        if self.observations:
            # Only keep last 3 observations to avoid bloat
            recent_obs = self.observations[-3:]
            parts.append(f"Key observations: {'; '.join(recent_obs)}")
        return " | ".join(parts)

@dataclass
class ConversationMemory:
    """Memory for the entire conversation, maintained locally"""
    completed_steps: List[StepMemory] = field(default_factory=list)
    current_step_id: str = ""
    active_context: Dict[str, str] = field(default_factory=dict)  # For cross-step data
    
    def add_step_memory(self, memory: StepMemory):
        """Add memory for a completed step"""
        self.completed_steps.append(memory)
        # Clean up old memories if too many (keep last 10)
        if len(self.completed_steps) > 10:
            # Summarize oldest steps into a compressed form
            self._compress_old_memories()
    
    def _compress_old_memories(self):
        """Compress oldest memories to prevent context bloat"""
        # Keep last 5 detailed, compress the rest
        to_compress = self.completed_steps[:-5]
        self.completed_steps = self.completed_steps[-5:]
        
        if to_compress:
            compressed = "Previous steps completed: " + ", ".join([
                f"{s.step_label} ({s.summary[:50] if s.summary else 'done'})" 
                for s in to_compress
            ])
            self.active_context["compressed_history"] = compressed
    
    def get_context_for_step(self, current_step: DAGNode, max_history: int = 5) -> str:
        """Get relevant context for the current step, limited to avoid bloat"""
        context_parts = []
        
        # Include compressed history if present
        if "compressed_history" in self.active_context:
            context_parts.append(self.active_context["compressed_history"])
        
        # Include recent relevant steps (same category or producing needed artifacts)
        relevant_steps = []
        for mem in reversed(self.completed_steps[-max_history:]):
            # Check if this step's artifacts are needed
            if current_step.dependencies:
                if any(dep in mem.artifacts for dep in current_step.dependencies):
                    relevant_steps.append(mem)
            # Or if same category for continuity
            elif mem.category == current_step.category:
                relevant_steps.append(mem)
        
        if relevant_steps:
            context_parts.append("Recent relevant steps:")
            for mem in relevant_steps[:3]:  # Limit to 3 most relevant
                context_parts.append(f"  • {mem.to_context_string()}")
        
        return "\n".join(context_parts) if context_parts else ""


# ----------------------------------------------------------------------
# DAG planning from the user goal
# ----------------------------------------------------------------------
def generate_dag(goal: str) -> Tuple[ProcessedDAG, List[str]]:
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

Deconstruct this statement into an actionable Directed Acyclic Graph (DAG) for an agentic LLM to finish the goal outlined in the input_statement.

## STEP 1: REASON ABOUT ORDERING AND CATEGORIZATION
Before outputting JSON, classify each step:
- **Reasoning steps**: Steps that require analysis, research, planning, or thinking
- **Actionable steps**: Steps that execute commands, create/modify files, install software
- **Verification steps**: Steps that check results, validate outcomes, ensure correctness
- **Decision steps**: Steps that require choosing between alternatives

For each step, identify:
- What files or data it depends on (dependencies)
- What files or data it produces (produces)
- What output is expected

## STEP 2: OUTPUT JSON
Return ONLY a valid JSON object with the following exact structure:

{{
  "reasoning": "Brief explanation of the logical order and why edges are correct",
  "dag": {{
    "nodes": [
      {{
        "id": "string", 
        "label": "string", 
        "type": "action|decision|start|end",
        "dependencies": ["file1", "data2"],  // What this step needs
        "produces": ["output.txt", "result.json"]  // What this step creates
      }}
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
2. **Self-loop rule**: No edge where from == to.
3. **Single end node**: Use exactly ONE node with type="end".
4. **Dependencies**: For actionable steps, list files/artifacts they need as dependencies.
5. **Produces**: For steps that create outputs, list what they produce.

Now process the input statement and return ONLY the JSON.
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
# Helper: Create memory from step execution
# ----------------------------------------------------------------------
def create_step_memory(
    step_node: DAGNode, 
    summary: str = "", 
    artifacts: List[str] = None,
    decisions: Dict[str, str] = None,
    observations: List[str] = None,
    exit_status: Optional[int] = None
) -> StepMemory:
    """Create a memory object for a completed step"""
    return StepMemory(
        step_id=step_node.id,
        step_label=step_node.label,
        category=step_node.category,
        summary=summary,
        artifacts=artifacts or [],
        decisions=decisions or {},
        observations=observations or [],
        exit_status=exit_status
    )


# ----------------------------------------------------------------------
# Agent loop for a single DAG node (sub‑task)
# ----------------------------------------------------------------------
def execute_agent_for_task(
    goal: str, 
    task_node: DAGNode, 
    memory: ConversationMemory,
    previous_outputs: Dict[str, str] = None
) -> StepMemory:
    """
    Run an interactive agent loop for one high‑level task.
    The agent proposes bash commands; the user must approve each one.
    Returns StepMemory with execution summary.
    """
    print(f"\n🤖 Working on: {task_node.label}")
    print(f"   Category: {task_node.category.value}")
    if task_node.dependencies:
        print(f"   Depends on: {', '.join(task_node.dependencies)}")
    if task_node.produces:
        print(f"   Produces: {', '.join(task_node.produces)}")

    # Build context-aware system prompt
    category_instructions = {
        NodeCategory.REASONING: """
This is a REASONING step. Your goal is to analyze, plan, or research.
You should:
- Focus on thinking through the problem
- Use commands like `ls`, `cat`, `find`, `grep` to inspect the environment
- Avoid creating or modifying files unless absolutely necessary
- Output your conclusions clearly for later steps
""",
        NodeCategory.ACTIONABLE: """
This is an ACTIONABLE step. Your goal is to execute commands that change the system.
You should:
- Create, modify, or delete files as needed
- Install packages, configure systems, run builds
- Use heredoc (<< 'EOF' ... EOF) for multiline content
- Avoid using 'echo' - prefer printf or cat with heredoc
""",
        NodeCategory.VERIFICATION: """
This is a VERIFICATION step. Your goal is to check that previous steps worked correctly.
You should:
- Use commands like `diff`, `test`, `grep -q` to validate
- Compare outputs to expected results
- Clearly state whether verification passed or failed
""",
        NodeCategory.DECISION: """
This is a DECISION step. Your goal is to choose between alternatives.
You should:
- Examine the current state using inspection commands
- Present options with pros/cons
- Ask the user for input if needed
- Document the decision made
"""
    }

    # Get relevant context from memory
    context_str = memory.get_context_for_step(task_node)
    
    # Get previous outputs that might be relevant
    deps_context = ""
    if previous_outputs and task_node.dependencies:
        relevant_outputs = {k: v for k, v in previous_outputs.items() 
                           if any(dep in k for dep in task_node.dependencies)}
        if relevant_outputs:
            deps_context = "\n\nOutputs from previous steps you should use:\n"
            for key, value in list(relevant_outputs.items())[:3]:  # Limit to 3
                deps_context += f"  {key}: {value[:200]}\n"

    system_prompt = (
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

        "VERIFICATION COMMANDS GUIDE:\n"
        "  ✓ Check file has content: `[ -s filename ] && echo 'HAS_CONTENT' || echo 'EMPTY'`\n"
        "  ✓ Count lines: `wc -l filename`\n"
        "  ✓ Preview content: `head -5 filename`\n"
        "  ✓ Check specific text: `grep -q 'pattern' filename && echo 'MATCH'`\n"
        "  ✗ AVOID: `test -f` (only checks existence, not content)\n"
        "  ✗ AVOID: commands that produce no output on success\n"
        "  IMPORTANT: Always use commands that give you actionable information!\n"
        "\n"
        
        "Be safe and avoid destructive commands without confirmation."
        + category_instructions.get(task_node.category, "")
        + """
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
    )

    user_prompt = (
        f"Overall goal: {goal}\n"
        f"Current subtask: {task_node.label}\n"
        f"Expected output: {task_node.expected_output or 'Not specified'}\n"
        f"Category: {task_node.category.value}\n"
        f"{deps_context}\n"
        f"{context_str}\n\n"
        "Start by assessing the situation and suggesting the first command.\n"
        "Remember: Use heredoc (<< 'EOF' ... EOF) for multiline content, avoid using 'echo'.\n"
        "IMPORTANT: Always end your command with 'END_COMMAND' on a new line.\n"
        "When complete, output 'DONE'."
    )

    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    done = False
    step_observations = []
    step_artifacts = []
    step_decisions = {}
    last_summary = ""

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

        # Extract command (same patterns as before)
        cmd = None
        
        pattern1 = r"COMMAND:\s*\n(.*?)\nEND_COMMAND"
        match1 = re.search(pattern1, content, re.DOTALL | re.MULTILINE)
        if match1:
            cmd = match1.group(1).strip()
        
        if not cmd:
            pattern2 = r"COMMAND:\s*(.*?)\nEND_COMMAND"
            match2 = re.search(pattern2, content, re.DOTALL | re.MULTILINE)
            if match2:
                cmd = match2.group(1).strip()
        
        if not cmd:
            pattern3 = r"COMMAND:\s*\n(.*?)(?=\n\n|$)"
            match3 = re.search(pattern3, content, re.DOTALL | re.MULTILINE)
            if match3:
                cmd = match3.group(1).strip()
                print("⚠️  Note: Command missing 'END_COMMAND' marker")
        
        if not cmd:
            single_line_match = re.search(r"^COMMAND:\s*(.*?)$", content, re.MULTILINE)
            if single_line_match:
                cmd = single_line_match.group(1).strip()
        
        if not cmd:
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
        
        # Extract reasoning
        reasoning = content.split("COMMAND:")[0].strip()
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
                        "content": "Command rejected. Please suggest an alternative command."
                    }
                )
                break

            elif choice in ("e", "edit"):
                print("Enter the new command (multiple lines allowed, press Ctrl+D or Ctrl+Z then Enter when done):")
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
            continue
        if done:
            break

        # Execute the approved command
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

        # Build result message
        result_msg = f"Exit code: {rc}\n"
        if stdout:
            result_msg += f"STDOUT:\n{stdout}\n"
        if stderr:
            result_msg += f"STDERR:\n{stderr}\n"
        if not stdout and not stderr:
            result_msg += "(no output)"

        conv.append({"role": "user", "content": f"Command result:\n{result_msg}"})
        print("📋 Output:\n" + result_msg)
        
        # Extract observations and artifacts from output
        if stdout:
            step_observations.append(stdout[:200])  # Store truncated observation
        
        # Try to detect created files from the command
        file_patterns = [r'cat > ([^\s]+)', r'printf.*> ([^\s]+)', r'cp ([^\s]+) ([^\s]+)']
        for pattern in file_patterns:
            matches = re.findall(pattern, cmd)
            for match in matches:
                if isinstance(match, tuple):
                    step_artifacts.extend([f for f in match if f.endswith(('.txt', '.json', '.py', '.md', '.log'))])
                else:
                    if match.endswith(('.txt', '.json', '.py', '.md', '.log')):
                        step_artifacts.append(match)
        step_artifacts = list(set(step_artifacts))[:5]  # Keep unique, max 5

    # After completion, ask for summary
    summary = input("What was achieved? (one sentence summary): ").strip()
    if not summary:
        summary = f"Completed {task_node.label}"
    
    # Ask for any key decisions made
    if task_node.category == NodeCategory.DECISION:
        decision = input("What decision was made? (press Enter to skip): ").strip()
        if decision:
            step_decisions["decision"] = decision
    
    return create_step_memory(
        step_node=task_node,
        summary=summary,
        artifacts=step_artifacts,
        decisions=step_decisions,
        observations=step_observations[:5]  # Keep last 5 observations
    )


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
        print("\n✅ Plan generated. Topological order:")
        
        # Show categorized order
        for i, node_id in enumerate(order, 1):
            node = dag.nodes[node_id]
            emoji = "🧠" if node.category == NodeCategory.REASONING else "⚡" if node.category == NodeCategory.ACTIONABLE else "✅" if node.category == NodeCategory.VERIFICATION else "🤔"
            print(f"  {i}. {emoji} {node.label} ({node.category.value})")
        
        # Initialize memory
        memory = ConversationMemory()
        previous_outputs = {}  # For tracking outputs between steps
        
        # Execute each node in order
        for i, node_id in enumerate(order, 1):
            node = dag.nodes[node_id]
            print(f"\n{'='*60}")
            print(f"Step {i}/{len(order)}: {node.label}")
            print(f"Category: {node.category.value.upper()}")
            print(f"{'='*60}")
            
            # For reasoning steps, we can provide a different prompt
            if node.category == NodeCategory.REASONING:
                print("💭 This is a reasoning step - focus on analysis and planning")
            
            step_memory = execute_agent_for_task(goal, node, memory, previous_outputs)
            memory.add_step_memory(step_memory)
            
            # Store outputs for dependency resolution
            if step_memory.artifacts:
                for artifact in step_memory.artifacts:
                    previous_outputs[artifact] = step_memory.summary
            if step_memory.summary:
                previous_outputs[node.id] = step_memory.summary
            
            print(f"✅ Step completed: {step_memory.summary}")

        print("\n🎉 Goal completed successfully!")
        
        # Print final memory summary
        print("\n📝 Execution Summary:")
        for mem in memory.completed_steps:
            print(f"  • {mem.step_label}: {mem.summary[:80]}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
