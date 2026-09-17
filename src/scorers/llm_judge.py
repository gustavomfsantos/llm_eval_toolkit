"""The evaluation engine. Uses an LLM to score another LLM's response - called
LLM-as-judge. Powerful but has a bias risk: the judge model carries its own
preferences. Using the same model as both subject and judge can create blind
spots where the judge systematically excuses its own failure patterns.
"""

import json

from logger import get_logger
from src.runner import call_ollama


logger = get_logger(__name__)


def build_judge_prompt(
    prompt: str,
    response: str,
    criterion: str,
    definition: str,
    rubric: dict,
) -> str:
    """Build a structured evaluation prompt for the local judge model.

    Args:
        prompt: Original question given to the evaluated model.
        response: Answer produced by the evaluated model.
        criterion: Name of the dimension being scored.
        definition: Plain-English definition of that dimension.
        rubric: Mapping of integer scores to their meanings.

    Returns:
        A complete prompt that requests a machine-readable JSON score.

    Example:
        >>> text = build_judge_prompt("2+2?", "4", "accuracy", "Correctness", {1: "wrong", 5: "right"})
        >>> 'accuracy' in text
        True
    """
    rubric_text = "\n".join(
        f"{score}: {description}" for score, description in sorted(rubric.items())
    )
    # Structured output is parseable. Free text would require regex and break
    # constantly.
    # Without a rubric, LLMs default to scoring 4-5 on almost everything. A
    # rubric anchors the scale.
    # Reasoning lets humans audit the judge. A score of 2 with no explanation is
    # useless for debugging.
    # Longer reasoning causes the model to second-guess its score mid-generation.
    return f"""You are an impartial evaluator of an AI-generated response.

Criterion: {criterion}
Definition: {definition}

Question:
{prompt}

Response:
{response}

Rubric:
{rubric_text}

Respond ONLY with valid JSON in exactly this shape:
{{"score": int, "reasoning": "one sentence"}}
Do not use Markdown, add extra keys, or include text outside the JSON object."""


def parse_judge_response(raw_response: str) -> dict:
    """Parse a JSON score from a raw Ollama response.

    Args:
        raw_response: Text returned by the judge model.

    Returns:
        A dictionary with integer ``score`` and string ``reasoning`` keys.
        Parse failures return a zero score and an explicit explanation.

    Example:
        >>> parse_judge_response('{"score": 5, "reasoning": "Correct."}')
        {'score': 5, 'reasoning': 'Correct.'}
    """
    # Local models are less reliable at producing clean JSON than frontier
    # models. Always strip backticks and handle parse failures.
    cleaned_response = raw_response.strip().replace("```json", "").replace("```", "")
    try:
        start = cleaned_response.index("{")
        end = cleaned_response.rindex("}") + 1
        parsed_response = json.loads(cleaned_response[start:end])
        return {
            "score": int(parsed_response["score"]),
            "reasoning": str(parsed_response["reasoning"]),
        }
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        logger.warning("Could not parse judge response '%s': %s", raw_response, error)
        return {"score": 0, "reasoning": "parse failed"}


def call_judge(
    prompt: str,
    response: str,
    criterion: str,
    definition: str,
    rubric: dict,
) -> dict:
    """Ask Ollama to score one response against a named rubric.

    Args:
        prompt: Original question given to the evaluated model.
        response: Answer produced by the evaluated model.
        criterion: Name of the dimension being scored.
        definition: Plain-English definition of that dimension.
        rubric: Mapping of integer scores to their meanings.

    Returns:
        A parsed dictionary with ``score`` and ``reasoning`` keys.

    Example:
        >>> result = call_judge("2+2?", "4", "accuracy", "Correctness", {1: "wrong", 5: "right"})
        >>> result["score"]
        5
    """
    judge_prompt = build_judge_prompt(
        prompt,
        response,
        criterion,
        definition,
        rubric,
    )
    raw_result = call_ollama(judge_prompt)
    parsed_result = parse_judge_response(raw_result.get("response_text", ""))
    logger.info(
        "Judge criterion '%s': score=%s, reasoning=%s",
        criterion,
        parsed_result["score"],
        parsed_result["reasoning"],
    )
    return parsed_result