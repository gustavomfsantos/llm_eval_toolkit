"""All communication with the local Ollama instance lives here. Isolated in one
file so swapping models or changing the API endpoint only requires editing this
file.
"""

import time

import pandas as pd
import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from logger import get_logger


logger = get_logger(__name__)


def call_ollama(prompt: str, context: str = "", model: str = "") -> dict:
    """Call the local Ollama generation endpoint and track request latency.

    Args:
        prompt: User question or instruction sent to the model.
        context: Optional context prepended to the question.
        model: Ollama model name, or the configured default when empty.

    Returns:
        A dictionary containing response text, latency in milliseconds, and an
        error string when the request fails.

    Example:
        >>> result = call_ollama("What is 2 + 2?")
        >>> "response_text" in result
        True
    """
    selected_model = model or OLLAMA_MODEL
    full_prompt = prompt
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    payload = {
        "model": selected_model,
        "prompt": full_prompt,
        # stream=False is critical here - without it Ollama returns tokens one
        # by one and the response field contains only the last token.
        "stream": False,
    }
    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        response_data = response.json()
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        # We track latency because local models vary widely in speed depending
        # on hardware. Logging it helps identify slow cases.
        logger.info("Ollama response completed in %d ms", latency_ms)
        return {
            "response_text": response_data["response"],
            "latency_ms": latency_ms,
        }
    except Exception as error:
        logger.error("Ollama request failed: %s", error)
        return {"response_text": "", "latency_ms": 0, "error": str(error)}


def run_batch(dataset: list[dict], model: str = "") -> pd.DataFrame:
    """Run Ollama once for each dataset case and collect tabular results.

    Args:
        dataset: Cases containing ``id``, ``prompt``, and optional ``context``.
        model: Ollama model name, or the configured default when empty.

    Returns:
        A DataFrame with prompts, contexts, expected answers, responses, and
        per-case latency values.

    Example:
        >>> results = run_batch([{"id": "one", "prompt": "Say hi"}])
        >>> list(results.columns)
        ['id', 'prompt', 'context', 'expected', 'response_text', 'latency_ms']
    """
    batch_start = time.perf_counter()
    rows = []
    total_cases = len(dataset)
    for index, case in enumerate(dataset, start=1):
        logger.info(
            "Running case %d/%d - id: %s",
            index,
            total_cases,
            case.get("id", "unknown"),
        )
        context = case.get("context", "") or ""
        result = call_ollama(case.get("prompt", ""), context=context, model=model)
        rows.append(
            {
                "id": case.get("id", ""),
                "prompt": case.get("prompt", ""),
                "context": context,
                "expected": case.get("expected", ""),
                "response_text": result.get("response_text", ""),
                "latency_ms": result.get("latency_ms", 0),
            }
        )

    total_time_ms = int((time.perf_counter() - batch_start) * 1000)
    total_latency = sum(row["latency_ms"] for row in rows)
    average_latency = total_latency / total_cases if total_cases else 0
    logger.info(
        "Batch completed in %d ms; average latency per case: %.1f ms",
        total_time_ms,
        average_latency,
    )
    # Sequential calls keep this simple for learning. With asyncio you could
    # run these in parallel and cut total time significantly.
    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "prompt",
            "context",
            "expected",
            "response_text",
            "latency_ms",
        ],
    )