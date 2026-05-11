#!/usr/bin/env python3
"""
llm_react_agent.py
Author: g023 - github.com/g023
License: MIT

Using the Qwen3-1.77B as a ReAct style agent that solves math problems.

Tested problem:
Q: "A merchant can fit 10 large boxes or 8 small boxes into a container for shipping. In one shipment, he sent 96 boxes. If there were more large boxes than small boxes, how many full containers did he ship?"
A: "11"

"""

import os
import json
import math

from typing import Dict, List, Any

from _inc_ollama import llm_stream

# ===

def _safer_calculate(expr: str) -> str:
    safe_dict = {
        'abs': abs, 'round': round,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'atan2': math.atan2, 'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'exp': math.exp, 'pi': math.pi, 'e': math.e,
    }
    allowed_chars = set('0123456789+-*/().% _,')
    if not all(c in allowed_chars or c.isalpha() for c in expr):
        return "Error: invalid characters"
    try:
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        if isinstance(result, float):
            result = round(result, 6)
        return f"{expr} = {result}"
    except Exception as e:
        return f"Math error: {e}"


# ===

class ReActAgent:
    """
    A ReAct agent following the DAG flow:
    Start -> Thought -> Action -> Observation -> Decision -> (Final Answer or Next Thought)
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.conversation_history = []  # stores full interaction for context
        self.max_iterations = 5  # safety to prevent infinite loops
        
    def _call_openai(self, prompt: str) -> str:
        """Unified OpenAI API call"""
        conv = []
        print("\n")

        print("v "*20)
        conv.append({"role":"user","content":prompt+"\n<ignore:no_think>\n"})
        res = llm_stream(conv)
        print("^ "*20)

        print("\n")
        return res["content"]

    
    def thought(self, question: str, observations_so_far: List[str]) -> str:
        """Generates a reasoning step (Thought node)"""
        obs_text = "\n".join(observations_so_far) if observations_so_far else "None yet."
        prompt = f"""You are an advanced step-by-step problem-solving and reasoning agent, 
that doesn't try to give the full answer on the first step. 

Did we make any mistakes? Yes? Then identify them, reconsider your plan at that node and regain your thought process. 
Self correction is the key strategy. 

Internally analyze the goal: identify core outcomes, hidden complexities, implicit constraints, and knowledge gaps.

Break each goal into the smallest practical, independent-yet-interdependent subtasks. 
Each subtask must:
- Have a single, verifiable outcome
- Be completable by an LLM agent in one reasoning pass and one tool call
- Include explicit success criteria
- note required knowledge
- note potential failure nodes

At each step, identify if we have strayed, what *eureka* moments you have, insights, *gotchas* and strategies for continuing to solve. You are a master problem solver that combines the intelligence of the deductive reasoners like Nikola Tesla, Albert Einstein, Steven Hawkings, Thomas Edison, etc. You will be critical and self-evaluating at every step. 

Once you've given the answer, self-verify and self-validate, by travelling backwards from the answer to the request and verify if it is the true answer. If at the end of any process, you've made an error, note the error for the next round to correct. Identify any problems with current logic and reasoning. 

Execute -> Reflect -> Self Correct

Your task is to think step by step about how to answer the user's question.

Consider the framing. Consider the elements of the problem. First understand the primary goal. Then identify any secondary goal. Is this a trick question? How should we break the problem up into sub problems to solve?

Question: {question}

Previous observations (from actions you took):
{obs_text}

Now, write your thought: what do you know, what are you missing, and what should you do next?
Keep it concise. Return just your 5 paragraphs of beautifully crafted thought rationale and nothing else.
"""
        return self._call_openai(prompt)
    
    def action(self, thought_text: str) -> Dict[str, Any]:
        """Decides which tool to use (Action node) and returns the action."""
        prompt = f"""You are an advanced step-by-step problem-solving and reasoning agent, that doesn't try to give the full answer on the first step. Your mission is to fulfill user requests using a systematic agentic methodology.


Based on the following thought, decide which tool to call.
Available tools:
- "search": use when you need external knowledge (e.g., facts, definitions).
- "calculate": use for arithmetic or math operations. Provide an expression.

Thought: {thought_text}

Respond in JSON format with keys: "tool", "input".
Example: {{"tool": "search", "input": "population of France"}}
Example: {{"tool": "calculate", "input": "25 * 4"}}
"""
        response = self._call_openai(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # fallback
            return {"tool": "search", "input": "default query"}
    
    def execute_action(self, action: Dict[str, Any]) -> str:
        """Simulate tool execution (Observation node)"""
        tool = action.get("tool")
        inp = action.get("input", "")
        
        if tool == "search":
            # For demo, return a canned response + simulated search result
            # In production, hook up to a real search API like Tavily, SerpAPI, etc.
            return f"[Search result for '{inp}']: The answer is approximately 42 (placeholder)."
        
        elif tool == "calculate":
            try:
                result = _safer_calculate(inp)  # caution: only for trusted expressions
                return f"[Calculation result]: {inp} = {result}"
            except Exception as e:
                return f"[Calculation error]: {str(e)}"
        
        else:
            return f"[Unknown tool '{tool}']: no result."
    
    def should_continue(self, question: str, last_observation: str, iteration: int) -> bool:
        """
        Decision node: based on the last observation, decide if answer is ready.
        If yes, we will go to Final Answer; otherwise we loop to a new Thought.
        """
        prompt = f"""
You are evaluating whether the agent has enough information to answer the user's question.
Question: {question}

Latest observation from the last action: {last_observation}

Current iteration count: {iteration} (max allowed is {self.max_iterations})

Answer ONLY with a JSON: {{"answer_ready": true/false, "reason": "short explanation"}}
Return true only if the observation clearly answers the question or if the agent has exhausted reasonable attempts.
"""
        response = self._call_openai(prompt)
        try:
            data = json.loads(response)
            return not data.get("answer_ready", False)  # if ready -> stop
        except:
            # default: continue if iteration is low
            return iteration < self.max_iterations
    
    def final_answer(self, question: str, observations: List[str], thoughts: List[str]) -> str:
        """Generate the final answer (Final Answer node)"""
        obs_text = "\n".join(observations)
        thought_text = "\n".join(thoughts)
        prompt = f"""You engage in socratic-insanity level of critical analysis of your past understandings and implicit knowledge. You call out perceived contradictions like they are begging to be called out. Draw connections and group data in logical clusters. Consider any past understandings and answer and conclude with what should be the right answer, regardless of what past answers perceived. 
        
Given the question and all evidence gathered, produce a clear, concise final answer. Return the final answer and nothing else.

Question: {question}

Thought process:
{thought_text}

Observations from actions:
{obs_text}

Final answer:
"""
        return self._call_openai(prompt)
    
    def run(self, question: str) -> str:
        """
        Execute the entire DAG flow externally: loops until answer_ready or max iterations.
        This corresponds to the "Continue? (external loop)" in the diagram.
        """
        print(f"\n🚀 Starting ReAct agent for question: {question}\n")
        
        observations = []
        thoughts = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")
            
            # Thought node
            current_thought = self.thought(question, observations)
            thoughts.append(current_thought)
            print(f"🧠 Thought: {current_thought}")
            
            # Action node
            action = self.action(current_thought)
            print(f"⚙️ Action: {action}")
            
            # Observation node
            observation = self.execute_action(action)
            observations.append(observation)
            print(f"👁️ Observation: {observation}")
            
            # Decision node
            continue_loop = self.should_continue(question, observation, iteration)
            if not continue_loop:
                print(f"✅ Decision: Answer ready. Moving to final answer.\n")
                break
            else:
                print(f"🔄 Decision: Not enough info. Continuing loop.\n")
        
        # Final Answer node (reached after loop exit)
        final = self.final_answer(question, observations, thoughts)
        print(f"📌 Final Answer: {final}")
        return final

# ========== Example usage ==========
if __name__ == "__main__":
    agent = ReActAgent() 
    question = """A merchant can fit 10 large boxes or 8 small boxes into a container for shipping. In one shipment, he sent 96 boxes. If there were more large boxes than small boxes, how many full containers did he ship?"""
    answer = agent.run(question)
