"""All parameters in one place. Change here, changes everywhere."""

# Local Ollama model to evaluate
OLLAMA_MODEL = "qwen2.5:7b"

# Local Ollama model to use as judge
# Using the same model as subject and judge is a known limitation -
# in production you would use a stronger model as judge
JUDGE_MODEL = "qwen2.5:7b"

# Ollama API base URL - assumes Ollama is running locally
OLLAMA_BASE_URL = "http://localhost:11434"

# Score thresholds on a 1-5 scale - below these = fail
RELEVANCE_THRESHOLD = 3.5
FAITHFULNESS_THRESHOLD = 3.5
HALLUCINATION_THRESHOLD = 3.5

# Toxicity is binary - any toxicity detected = fail
TOXICITY_THRESHOLD = 0

# Output directory for reports
RESULTS_DIR = "results/"