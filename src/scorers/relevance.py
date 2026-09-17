"""Scores whether the response answers the question asked. A response can be
factually correct and still be irrelevant if it answers a different question.
"""

import pandas as pd

from config import RELEVANCE_THRESHOLD
from logger import get_logger
from src.scorers.llm_judge import call_judge


logger = get_logger(__name__)


def score_relevance(prompt: str, response: str) -> dict:
    """Score how directly a response addresses its original question.

    Args:
        prompt: Question sent to the evaluated model.
        response: Answer produced by the evaluated model.

    Returns:
        A score, judge reasoning, and threshold-based pass flag.

    Example:
        >>> result = score_relevance("What is 2+2?", "4")
        >>> set(result) == {"score", "reasoning", "passed"}
        True
    """
    rubric = {
        1: "The response does not address the question.",
        3: "The response partially answers the question or wanders.",
        5: "The response fully and directly answers the question.",
    }
    result = call_judge(
        prompt,
        response,
        "relevance",
        "Whether the response answers the question that was asked.",
        rubric,
    )
    return {
        "score": result["score"],
        "reasoning": result["reasoning"],
        "passed": result["score"] >= RELEVANCE_THRESHOLD,
    }


def score_relevance_batch(results_df: pd.DataFrame) -> pd.DataFrame:
    """Score relevance for every row in a model-results DataFrame.

    Args:
        results_df: DataFrame with ``prompt`` and ``response_text`` columns.

    Returns:
        A copy with relevance score, reasoning, and pass columns added.

    Example:
        >>> frame = pd.DataFrame([{"prompt": "Q", "response_text": "A"}])
        >>> "relevance_passed" in score_relevance_batch(frame)
        True
    """
    scored_results = results_df.copy()
    scores = []
    reasonings = []
    passed_values = []
    for index, row in scored_results.iterrows():
        logger.info("Scoring relevance case %s", row.get("id", index))
        result = score_relevance(row.get("prompt", ""), row.get("response_text", ""))
        scores.append(result["score"])
        reasonings.append(result["reasoning"])
        passed_values.append(result["passed"])
    scored_results["relevance_score"] = scores
    scored_results["relevance_reasoning"] = reasonings
    scored_results["relevance_passed"] = passed_values
    pass_rate = sum(passed_values) / len(passed_values) if passed_values else 0
    logger.info("Relevance pass rate: %.1f%%", pass_rate * 100)
    return scored_results