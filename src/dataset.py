"""Loads and validates eval datasets. Validation happens before any model calls -
a malformed test case would silently corrupt eval results downstream.
"""

import json
from pathlib import Path

from logger import get_logger


logger = get_logger(__name__)


def load_dataset(path: str) -> list[dict]:
    """Load a JSON dataset and warn about missing schema fields.

    Args:
        path: Path to a dataset JSON file. The filename selects its schema.

    Returns:
        A list of dataset records, including records with validation warnings.

    Example:
        >>> cases = load_dataset("datasets/factual_qa.json")
        >>> len(cases)
        20
    """
    dataset_path = Path(path)
    dataset_type = dataset_path.stem
    required_fields = {
        "factual_qa": {"id", "prompt", "expected", "category"},
        "rag_eval": {"id", "context", "prompt", "expected", "category"},
        "safety_prompts": {"id", "prompt", "expected_behavior", "category"},
    }.get(dataset_type, set())

    with dataset_path.open(encoding="utf-8") as file_handle:
        dataset = json.load(file_handle)

    if not isinstance(dataset, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}")

    for case in dataset:
        if not isinstance(case, dict):
            logger.warning("Dataset %s contains a non-object case", dataset_path)
            continue
        missing_fields = required_fields.difference(case)
        if missing_fields:
            logger.warning(
                "Dataset %s case %s is missing fields: %s",
                dataset_path,
                case.get("id", "unknown"),
                ", ".join(sorted(missing_fields)),
            )

    logger.info(
        "Loaded %d cases from %s dataset: %s",
        len(dataset),
        dataset_type,
        dataset_path,
    )
    return dataset


def preview_dataset(dataset: list[dict], n: int = 3) -> None:
    """Log the first few dataset cases in a readable format.

    Args:
        dataset: Dataset records to preview.
        n: Maximum number of records to log.

    Returns:
        None.

    Example:
        >>> preview_dataset([{"id": "case_1", "prompt": "Hello"}], n=1)
    """
    # Always preview before running - catching a bad case here saves inference time.
    logger.info(
        "Dataset preview: showing %d of %d cases",
        min(n, len(dataset)),
        len(dataset),
    )
    for case in dataset[:n]:
        logger.info("Case %s: %s", case.get("id", "unknown"), case)


def filter_by_category(dataset: list[dict], category: str) -> list[dict]:
    """Return dataset cases that belong to one category.

    Args:
        dataset: Dataset records to filter.
        category: Category name to match exactly.

    Returns:
        A list containing only records with the requested category.

    Example:
        >>> filter_by_category([{"category": "science"}], "science")
        [{'category': 'science'}]
    """
    filtered_dataset = [
        case for case in dataset if case.get("category") == category
    ]
    logger.info(
        "Category '%s': matched %d of %d cases",
        category,
        len(filtered_dataset),
        len(dataset),
    )
    return filtered_dataset