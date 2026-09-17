"""Generates reports in two formats: JSON for machines, HTML for humans. The
HTML report is what you show stakeholders - color-coded by pass/fail so the
story is immediate.
"""

import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import OLLAMA_MODEL
from logger import get_logger


logger = get_logger(__name__)


def generate_json_report(
    summary: dict,
    scored_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write summary metrics and detailed scores as a JSON report.

    Args:
        summary: Aggregated evaluation metrics.
        scored_df: Detailed per-case scorer results.
        output_path: Destination path for the JSON file.

    Returns:
        None.

    Example:
        >>> generate_json_report({}, pd.DataFrame(), "results/example.json")
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": json.loads(scored_df.to_json(orient="records")),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "JSON report saved to %s (%d bytes)",
        report_path,
        report_path.stat().st_size,
    )


def generate_html_report(
    summary: dict,
    scored_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write a standalone HTML report with summary cards and detail rows.

    Args:
        summary: Aggregated evaluation metrics.
        scored_df: Detailed per-case scorer results.
        output_path: Destination path for the HTML file.

    Returns:
        None.

    Example:
        >>> generate_html_report({}, pd.DataFrame(), "results/example.html")
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cards = []
    for dimension, values in summary.items():
        if not isinstance(values, dict) or "pass_rate" not in values:
            continue
        pass_rate = float(values["pass_rate"])
        color_class = (
            "green" if pass_rate >= 0.8 else "yellow" if pass_rate >= 0.6 else "red"
        )
        cards.append(
            f"""<section class="card {color_class}">
                <h2>{html.escape(dimension.title())}</h2>
                <strong>{pass_rate:.1%}</strong>
                <span>Pass rate · mean score {values.get('mean_score', 0):.2f}</span>
                <span>{values.get('n', 0)} cases scored</span>
            </section>"""
        )

    detail_rows = []
    score_dimensions = [
        ("relevance", "relevance_score", "relevance_passed"),
        ("faithfulness", "faithfulness_score", "faithfulness_passed"),
        ("hallucination", "hallucination_score", "hallucination_passed"),
    ]
    for _, row in scored_df.iterrows():
        score_cells = []
        for dimension, score_column, passed_column in score_dimensions:
            score = row.get(score_column)
            passed = row.get(passed_column)
            cell_class = "pass" if pd.notna(passed) and bool(passed) else "fail"
            display_score = "-" if pd.isna(score) else f"{float(score):.1f}"
            score_cells.append(
                f'<td class="{cell_class}">{dimension.title()}: {display_score}</td>'
            )
        if "toxicity_flag" in row.index:
            toxic = bool(row.get("toxicity_flag", False))
            score_cells.append(
                f'<td class="{"fail" if toxic else "pass"}">'
                f"Toxicity: {'FLAG' if toxic else 'Clear'}</td>"
            )
        detail_rows.append(
            f"<tr><td>{html.escape(str(row.get('id', '')))}</td>"
            f"<td>{html.escape(str(row.get('prompt', '')))}</td>"
            f"{''.join(score_cells)}</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Evaluation Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #17202a; }}
header {{ border-bottom: 3px solid #17202a; margin-bottom: 1.5rem; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }}
.card {{ border-left: 6px solid; padding: 1rem; min-width: 170px;
background: #f5f6f7; }}
.card.green {{ border-color: #238636; }}
.card.yellow {{ border-color: #b08800; }}
.card.red {{ border-color: #cf222e; }}
.card h2 {{ margin: 0 0 .5rem; font-size: 1rem; }}
.card strong {{ display: block; font-size: 1.8rem; }}
.card span {{ display: block; color: #57606a; font-size: .85rem; margin-top: .3rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
th, td {{ border: 1px solid #d0d7de; padding: .6rem; text-align: left; }}
th {{ background: #17202a; color: white; }}
td.pass {{ background: #dafbe1; }}
td.fail {{ background: #ffebe9; }}
</style>
</head>
<body>
<header><h1>LLM Evaluation Report</h1>
<p>Model: {html.escape(OLLAMA_MODEL)} · Generated: {report_date} ·
Total cases: {len(scored_df)}</p></header>
<main><div class="cards">{''.join(cards)}</div>
<h2>Case details</h2>
<table><thead><tr><th>ID</th><th>Prompt</th><th>Relevance</th>
<th>Faithfulness</th><th>Hallucination</th><th>Toxicity</th></tr></thead>
<tbody>{''.join(detail_rows)}</tbody></table>
</main>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    logger.info("HTML report saved to %s", report_path)


def print_console_summary(summary: dict) -> None:
    """Log an ASCII scoreboard for each summarized dimension.

    Args:
        summary: Aggregated evaluation metrics keyed by dimension.

    Returns:
        None.

    Example:
        >>> print_console_summary(
        ...     {"relevance": {"mean_score": 4.5, "pass_rate": 1.0, "n": 2}}
        ... )
    """
    logger.info("=" * 60)
    logger.info("LLM EVALUATION SCOREBOARD")
    for dimension, values in summary.items():
        if not isinstance(values, dict) or "pass_rate" not in values:
            continue
        status = "PASS" if values["pass_rate"] >= 0.8 else "FAIL"
        logger.info(
            "%-16s score %.2f | pass rate %6.1f%% | %s",
            dimension,
            values.get("mean_score", 0.0),
            values["pass_rate"] * 100,
            status,
        )