# llm_eval_toolkit

An educational Python framework for measuring the quality of responses from a
locally running Ollama model. It uses `qwen2.5:7b` through Ollama's local HTTP
API, so it requires no paid API account.

## Flow

```text
Dataset -> Runner -> Scorers -> Aggregator -> Report
```

## Evaluation dimensions

| Dimension | What it measures |
| --- | --- |
| Relevance | Whether the response answers the question asked |
| Faithfulness | Whether a RAG response stays within supplied context |
| Hallucination | Whether a response agrees with known ground truth |
| Toxicity | Whether a response contains toxic or dangerous content |

The same local model is used as both subject and judge in this educational
project. That is convenient for learning but creates evaluator bias and blind
spots; production systems should use an independent, stronger judge.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
python main.py
```

Ollama must be running at `http://localhost:11434`. The pipeline writes
`results/report.json` for machines and `results/report.html` for people.

## Tech stack

Python · Requests · Pandas · Ollama · qwen2.5:7b

## Project structure

- `config.py` keeps model names, thresholds, and paths together.
- `src/dataset.py` validates cases before inference.
- `src/runner.py` isolates local Ollama communication and latency tracking.
- `src/scorers/` contains focused scoring functions.
- `src/aggregator.py` summarizes scores and flags the worst cases.
- `src/reporting.py` produces JSON and single-file HTML reports.
- `notebooks/exploration.ipynb` explains the pipeline interactively.