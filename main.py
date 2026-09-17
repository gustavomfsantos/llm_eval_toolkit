"""Entry point. Runs the full evaluation pipeline in five stages. Each stage is
clearly logged so you can follow what is happening and why.
"""

from config import OLLAMA_MODEL, RESULTS_DIR
from logger import get_logger
from src.aggregator import (
    compute_dimension_summary,
    compute_overall_pass_rate,
    flag_worst_cases,
)
from src.dataset import load_dataset, preview_dataset
from src.reporting import (
    generate_html_report,
    generate_json_report,
    print_console_summary,
)
from src.runner import run_batch
from src.scorers.hallucination import score_hallucination_batch
from src.scorers.relevance import score_relevance_batch
from src.scorers.toxicity import score_toxicity_batch


logger = get_logger(__name__)


def main() -> None:
    """Run the complete factual QA evaluation pipeline.

    Args:
        None.

    Returns:
        None.

    Example:
        >>> main()
    """
    logger.info("=" * 60)
    logger.info("Stage 1/5: Load dataset")
    dataset = load_dataset("datasets/factual_qa.json")
    preview_dataset(dataset, n=3)

    logger.info("=" * 60)
    logger.info("Stage 2/5: Run model batch")
    estimated_seconds = len(dataset) * 10
    logger.info(
        "Estimated runtime: about %.1f minutes for %d cases",
        estimated_seconds / 60,
        len(dataset),
    )
    results_df = run_batch(dataset, model=OLLAMA_MODEL)

    logger.info("=" * 60)
    logger.info("Stage 3/5: Score all outputs")
    scored_df = score_relevance_batch(results_df)
    scored_df = score_hallucination_batch(scored_df)
    scored_df = score_toxicity_batch(scored_df)

    logger.info("=" * 60)
    logger.info("Stage 4/5: Aggregate results")
    summary = compute_dimension_summary(scored_df)
    summary["overall_pass_rate"] = compute_overall_pass_rate(scored_df)
    flag_worst_cases(scored_df, n=5)

    logger.info("=" * 60)
    logger.info("Stage 5/5: Generate reports")
    generate_json_report(summary, scored_df, f"{RESULTS_DIR}/report.json")
    generate_html_report(summary, scored_df, f"{RESULTS_DIR}/report.html")
    print_console_summary(summary)
    logger.info("Pipeline complete. Open results/report.html to view full results.")


if __name__ == "__main__":
    main()