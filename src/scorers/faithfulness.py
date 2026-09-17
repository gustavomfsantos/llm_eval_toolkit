"""Scores whether the response stays within the provided context. Critical for
RAG - a model that ignores context and uses its own knowledge is unpredictable
in production.
"""

import pandas as pd

from config import FAITHFULNESS_THRESHOLD
from logger import get_logger
from src.scorers.llm_judge import call_judge


logger = get_logger(__name__)


def score_faithfulness(context: str, response: str) -> dict:
    """Score whether a response is supported by the supplied context.

    Args:
        context: Reference material available to the evaluated model.
        response: Answer produced by the evaluated model.

    Returns:
        A score, reasoning, and pass flag, or ``None`` values when context is
        absent and faithfulness is not applicable.

    Example:
        >>> score_faithfulness("The sky is blue.", "The sky is blue.")["passed"]
        True
    """
    if not context:
        return {
            "score": None,
            "reasoning": "no context - not applicable",
            "passed": None,
        }
    rubric = {
        1: "The response contradicts the context.",
        3: "The response is mostly faithful but adds unsupported claims.",
        5: "Every claim in the response is supported by the context.",
    }
    result = call_judge(
        context,
        response,
        "faithfulness",
        "Whether every claim in the response is supported by the provided context.",
        rubric,
    )
    return {
        "score": result["score"],
        "reasoning": result["reasoning"],
        "passed": result["score"] >= FAITHFULNESS_THRESHOLD,
    }


def score_faithfulness_batch(results_df: pd.DataFrame) -> pd.DataFrame:
    """Score faithfulness for rows that have reference context.

    Args:
        results_df: DataFrame with ``context`` and ``response_text`` columns.

    Returns:
        A copy with faithfulness score, reasoning, and pass columns added.

    Example:
        >>> frame = pd.DataFrame([{"context": "C", "response_text": "A"}])
        >>> "faithfulness_score" in score_faithfulness_batch(frame)
        True
    """
    scored_results = results_df.copy()
    scores = []
    reasonings = []
    passed_values = []
    for index, row in scored_results.iterrows():
        result = score_faithfulness(
            row.get("context", "") or "",
            row.get("response_text", ""),
        )
        logger.info("Scoring faithfulness case %s", row.get("id", index))
        scores.append(result["score"])
        reasonings.append(result["reasoning"])
        passed_values.append(result["passed"])
    scored_results["faithfulness_score"] = scores
    scored_results["faithfulness_reasoning"] = reasonings
    scored_results["faithfulness_passed"] = passed_values
    applicable = [value for value in passed_values if value is not None]
    pass_rate = sum(applicable) / len(applicable) if applicable else 0
    logger.info("Faithfulness pass rate: %.1f%%", pass_rate * 100)
    return scored_results