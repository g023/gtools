<p align="center">
  <b>⚡ Tiny Model. Giant Possibilities. ⚡</b><br>
  A local‑first agentic toolkit powered entirely by a Qwen3 1.77B parameter LLM running on your machine.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-100%25-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/g023/gtools/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/Powered%20by-Ollama-000000?logo=ollama" alt="Ollama"></a>
  <a href="https://huggingface.co/g023/Qwen3-1.77B-g023-GGUF"><img src="https://img.shields.io/badge/Model-Qwen3--1.77B--g023-ff6f00?logo=huggingface" alt="Model"></a>
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" alt="Status">
</p>

---

## 🧠 Why g023's Local AI Agentic gTools?

**g023's Local AI Agentic gTools** is an experimental LLM toolbox. It’s minimal, elegant, and *fully local* agentic tooling and examples that should make it easy to apply/adapt to other uses. Every tool/example runs entirely on your hardware, driven by a single **Qwen3 1.77 billion parameter model** that is sufficient for some degree of reasoning and tool use.

No cloud. No API keys. No hidden costs. Just a tiny, blazing‑fast model and some python tools to show how far small models can go.

**ALL AGENTIC TOOLS USE https://huggingface.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧩 **Modular Toolkit** | Independent, reusable Python modules for building/learning/understanding/improving agentic systems. |
| 🏠 **100% Local** | Runs entirely on your machine via Ollama—privacy‑first, low latency. |

---

## 🚀 Quick Start

### Prerequisites
- [Ollama](https://ollama.com) installed and running.
- Python 3.10+.
- https://huggingface.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0 installed in ollama:
  ```
  ollama pull hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0
  ```

### Installation
```bash
git clone https://github.com/g023/gtools.git
cd gtools
```

### Standalone Examples:
```
python3 llm_bash_agent_simple.py 

python3 llm_bash_agent_advanced.py 

python3 llm_judge_test.py 

python3 llm_react_agent.py
```
