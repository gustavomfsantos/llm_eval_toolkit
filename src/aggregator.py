"""Combines individual scorer outputs into a summary. Think of this as the
scoreboard - it answers: how did the model do overall, and where did it struggle
most?
"""

import pandas as pd

from logger import get_logger


logger = get_logger(__name__)


def compute_dimension_summary(scored_df: pd.DataFrame) -> dict:
    """Compute mean score, pass rate, and case count for each dimension.

    Args:
        scored_df: DataFrame containing scorer output columns.

    Returns:
        A dictionary keyed by dimension with mean score, pass rate, and n.

    Example:
        >>> frame = pd.DataFrame({"relevance_score": [5], "relevance_passed": [True]})
        >>> compute_dimension_summary(frame)["relevance"]["n"]
        1
    """
    dimensions = {
        "relevance": ("relevance_score", "relevance_passed"),
        "faithfulness": ("faithfulness_score", "faithfulness_passed"),
        "hallucination": ("hallucination_score", "hallucination_passed"),
    }
    summary = {}
    for dimension, (score_column, passed_column) in dimensions.items():
        if score_column not in scored_df or passed_column not in scored_df:
            continue
        scores = pd.to_numeric(scored_df[score_column], errors="coerce").dropna()
        passed = scored_df[passed_column].dropna().astype(bool)
        summary[dimension] = {
            "mean_score": float(scores.mean()) if not scores.empty else 0.0,
            "pass_rate": float(passed.mean()) if not passed.empty else 0.0,
            "n": int(len(scores)),
        }

    if "toxicity_flag" in scored_df:
        toxicity_flags = scored_df["toxicity_flag"].dropna().astype(bool)
        safe_scores = (~toxicity_flags).astype(float)
        summary["toxicity"] = {
            "mean_score": float(safe_scores.mean()) if not safe_scores.empty else 0.0,
            "pass_rate": float(safe_scores.mean()) if not safe_scores.empty else 0.0,
            "n": int(len(safe_scores)),
        }

    for dimension, values in summary.items():
        logger.info(
            "%s: mean=%.2f, pass rate=%.1f%%, n=%d",
            dimension,
            values["mean_score"],
            values["pass_rate"] * 100,
            values["n"],
        )
    return summary


def compute_overall_pass_rate(scored_df: pd.DataFrame) -> float:
    """Compute the share of cases passing every applicable dimension.

    Args:
        scored_df: DataFrame containing scorer pass/fail columns.

    Returns:
        Overall pass rate as a float between 0 and 1.

    Example:
        >>> frame = pd.DataFrame({"relevance_passed": [True, False]})
        >>> compute_overall_pass_rate(frame)
        0.5
    """
    pass_columns = [
        "relevance_passed",
        "faithfulness_passed",
        "hallucination_passed",
    ]
    applicable_columns = [
        column for column in pass_columns if column in scored_df.columns
    ]
    if "toxicity_flag" in scored_df:
        scored_df = scored_df.copy()
        scored_df["toxicity_passed"] = ~scored_df["toxicity_flag"].astype(bool)
        applicable_columns.append("toxicity_passed")

    passed_cases = 0
    scored_cases = 0
    for _, row in scored_df.iterrows():
        applicable_values = [
            bool(row[column])
            for column in applicable_columns
            if pd.notna(row[column])
        ]
        if applicable_values:
            scored_cases += 1
            if all(applicable_values):
                passed_cases += 1
    pass_rate = passed_cases / scored_cases if scored_cases else 0.0
    logger.info(
        "Overall pass rate: %.1f%% (%d/%d cases)",
        pass_rate * 100,
        passed_cases,
        scored_cases,
    )
    # AND logic is strict - one failure fails the whole case. Some teams weight
    # dimensions differently. The right choice depends on what failure costs.
    return pass_rate


def flag_worst_cases(scored_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return cases with the lowest aggregate dimension scores.

    Args:
        scored_df: DataFrame containing scorer score and flag columns.
        n: Maximum number of worst cases to return.

    Returns:
        A DataFrame containing the worst cases in ascending aggregate order.

    Example:
        >>> frame = pd.DataFrame({"id": ["a"], "relevance_score": [1]})
        >>> len(flag_worst_cases(frame, n=1))
        1
    """
    score_columns = [
        column
        for column in [
            "relevance_score",
            "faithfulness_score",
            "hallucination_score",
        ]
        if column in scored_df.columns
    ]
    ranked_results = scored_df.copy()
    score_frame = ranked_results[score_columns].apply(pd.to_numeric, errors="coerce")
    if "toxicity_flag" in ranked_results:
        score_frame["toxicity_score"] = (~ranked_results["toxicity_flag"].astype(bool)).astype(float) * 5
    ranked_results["_aggregate_score"] = score_frame.mean(axis=1, skipna=True)
    worst_cases = ranked_results.sort_values("_aggregate_score").head(n).drop(
        columns=["_aggregate_score"]
    )
    # Worst cases are more valuable than average metrics. They show exactly
    # where and how the model fails.
    for _, case in worst_cases.iterrows():
        logger.info("Worst case %s: %s", case.get("id", "unknown"), case.to_dict())
    return worst_cases