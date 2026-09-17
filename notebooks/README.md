# Notebook walkthrough

Open `exploration.ipynb` after installing the project requirements and starting
Ollama with `qwen2.5:7b` available locally. The notebook follows the same five
stages as `main.py`, but leaves each intermediate DataFrame visible so you can
inspect what changes after loading, generation, scoring, aggregation, and
reporting.

The notebook makes real local Ollama calls in stages 2 and 3. Run those cells
only when the local model server is available.