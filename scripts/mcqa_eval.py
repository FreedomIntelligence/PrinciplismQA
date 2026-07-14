#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible model on PrinciplismQA multiple-choice items.

Every completed record preserves the model's raw ``original_response``. The
script extracts an A-D ``model_answer`` only when one is unambiguous. Refusals
and unparseable replies are retained and reported separately rather than being
silently treated as incorrect answers.

Examples:
  # Validate and inspect three questions without contacting a model.
  python scripts/mcqa_eval.py --dry-run --limit 3

  # Run five questions, checkpointing each completed result.
  MCQA_API_KEY=... python scripts/mcqa_eval.py \
    --model gpt-4o --output results/mcqa_eval.json --limit 5 --save-interval 1

  # Analyze a previous result without calling an API.
  python scripts/mcqa_eval.py --analyze results/mcqa_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from openai import OpenAI
from tqdm import tqdm


REPOSITORY_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPOSITORY_DIR / "data" / "knowledge-mcqa.json"
VALID_ANSWERS = {"A", "B", "C", "D"}
REFUSAL_PATTERN = re.compile(
    r"\b(?:i\s+(?:cannot|can['’]t|am\s+unable|won['’]t)|"
    r"i\s+(?:must|have\s+to)\s+decline|"
    r"(?:cannot|can['’]t|unable\s+to)\s+(?:answer|assist|help|provide)|"
    r"not\s+(?:able|permitted|comfortable)\s+to\s+(?:answer|assist|help|provide)|"
    r"(?:as\s+an\s+ai|i\s+do\s+not)\s+(?:cannot|can['’]t|won['’]t))\b",
    flags=re.IGNORECASE,
)
DIRECT_ANSWER_PATTERN = re.compile(r"^\s*[\[(]?([A-D])[\])]?\s*(?:[.:;!,-].*)?$", re.IGNORECASE | re.DOTALL)
MARKED_ANSWER_PATTERN = re.compile(
    r"\b(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|option|choice)"
    r"\s*(?:is|:|-)?\s*[\[(]?([A-D])[\])]?\b",
    flags=re.IGNORECASE,
)
FIRST_LINE_ANSWER_PATTERN = re.compile(r"^\s*[\[(]?([A-D])[\])]?\s*[.)-]\s+", flags=re.IGNORECASE)

PROMPT_TEMPLATE = """Answer this medical ethics multiple-choice question.

Select the single best option. Respond with only one letter: A, B, C, or D. Do not provide an explanation.

Question:
{question}

Options:
{options}
"""


@dataclass(frozen=True)
class McqaItem:
    item_id: int
    question_id: int
    question: str
    options: dict[str, str]
    correct_answer: str
    principlism: dict[str, bool]


@dataclass(frozen=True)
class EndpointConfig:
    model: str
    api_key_env: str
    base_url: str | None


def read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return value


def load_items(path: Path) -> list[McqaItem]:
    """Load and validate the MCQA data required by this evaluator."""
    items: list[McqaItem] = []
    question_ids: set[int] = set()
    for raw_item in read_json_array(path):
        item_id = raw_item.get("id")
        question_id = raw_item.get("question_id")
        question = raw_item.get("question")
        options = raw_item.get("options")
        correct_answer = raw_item.get("correct_answer")
        principlism = raw_item.get("principlism")
        if (
            not isinstance(item_id, int)
            or not isinstance(question_id, int)
            or not isinstance(question, str)
            or not isinstance(options, dict)
            or not isinstance(correct_answer, str)
            or not isinstance(principlism, dict)
        ):
            raise ValueError("Each MCQA record must contain id, question_id, question, options, correct_answer, and principlism")
        if question_id in question_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        question_ids.add(question_id)
        if set(options) != VALID_ANSWERS or not all(isinstance(value, str) for value in options.values()):
            raise ValueError(f"Question {question_id} must have exactly A-D string options")
        if correct_answer not in VALID_ANSWERS:
            raise ValueError(f"Question {question_id} has invalid correct_answer {correct_answer!r}")
        if not all(isinstance(name, str) and isinstance(value, bool) for name, value in principlism.items()):
            raise ValueError(f"Question {question_id} has invalid principlism annotations")
        items.append(
            McqaItem(
                item_id=item_id,
                question_id=question_id,
                question=question,
                options={answer: options[answer] for answer in sorted(VALID_ANSWERS)},
                correct_answer=correct_answer,
                principlism=dict(principlism),
            )
        )
    return sorted(items, key=lambda item: item.question_id)


def load_output(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in output file {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Output file {path} must use this script's object-with-items format")
    records: dict[int, dict[str, Any]] = {}
    for record in payload["items"]:
        if not isinstance(record, dict) or not isinstance(record.get("question_id"), int):
            raise ValueError(f"Output file {path} contains an invalid record")
        question_id = record["question_id"]
        if question_id in records:
            raise ValueError(f"Output file {path} contains duplicate question_id {question_id}")
        records[question_id] = record
    return records


def write_output(path: Path, records: Iterable[dict[str, Any]], config: EndpointConfig) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "items": sorted(records, key=lambda record: record["question_id"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2)
            target.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def get_client(config: EndpointConfig) -> OpenAI:
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"Set {config.api_key_env} before calling model {config.model}")
    return OpenAI(api_key=api_key, base_url=config.base_url or None, timeout=180.0)


def call_model(client: OpenAI, config: EndpointConfig, prompt: str, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Model returned no message content")
            return response.choices[0].message.content
        except Exception as error:  # Compatible API clients expose different error classes.
            last_error = error
            if attempt == retries - 1:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"API call failed after {retries} attempts: {last_error}") from last_error


def extract_answer(original_response: str) -> tuple[str, str | None]:
    """Classify a response and extract a letter only when it is unambiguous."""
    cleaned = original_response.strip()
    if not cleaned:
        return "unparseable", None
    direct_match = DIRECT_ANSWER_PATTERN.fullmatch(cleaned)
    if direct_match:
        return "answered", direct_match.group(1).upper()
    first_line_match = FIRST_LINE_ANSWER_PATTERN.match(cleaned)
    if first_line_match:
        return "answered", first_line_match.group(1).upper()
    marked_answers = MARKED_ANSWER_PATTERN.findall(cleaned)
    if marked_answers:
        unique_answers = {answer.upper() for answer in marked_answers}
        if len(unique_answers) == 1:
            return "answered", unique_answers.pop()
    if REFUSAL_PATTERN.search(cleaned):
        return "refusal", None
    return "unparseable", None


def base_record(item: McqaItem) -> dict[str, Any]:
    return {
        "id": item.item_id,
        "question_id": item.question_id,
        "question": item.question,
        "correct_answer": item.correct_answer,
        "principlism": item.principlism,
    }


def record_matches_item(record: dict[str, Any] | None, item: McqaItem) -> bool:
    return bool(
        record
        and record.get("id") == item.item_id
        and record.get("question") == item.question
        and record.get("correct_answer") == item.correct_answer
        and record.get("principlism") == item.principlism
    )


def record_is_reusable(record: dict[str, Any] | None, item: McqaItem, model: str) -> bool:
    return bool(
        record_matches_item(record, item)
        and record is not None
        and record.get("model") == model
        and isinstance(record.get("original_response"), str)
        and record.get("response_status") in {"answered", "refusal", "unparseable"}
    )


def evaluate_item(
    item: McqaItem,
    prior_record: dict[str, Any] | None,
    client: OpenAI,
    config: EndpointConfig,
    retries: int,
) -> dict[str, Any]:
    options = "\n".join(f"{letter}. {text}" for letter, text in item.options.items())
    original_response = call_model(
        client,
        config,
        PROMPT_TEMPLATE.format(question=item.question, options=options),
        retries,
    )
    response_status, model_answer = extract_answer(original_response)
    record = base_record(item)
    record.update(
        {
            "model": config.model,
            "original_response": original_response,
            "response_status": response_status,
            "model_answer": model_answer,
            "is_correct": model_answer == item.correct_answer if model_answer is not None else None,
        }
    )
    return record


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build accuracy, refusal, and principle-level statistics from a result JSON."""
    materialized_records = list(records)
    counts = {"answered": 0, "correct": 0, "incorrect": 0, "refusal": 0, "unparseable": 0}
    principles: dict[str, dict[str, int]] = {}
    models: set[str] = set()

    for record in materialized_records:
        if not isinstance(record, dict):
            raise ValueError("Result items must be objects")
        response = record.get("original_response")
        status = record.get("response_status")
        model_answer = record.get("model_answer")
        is_correct = record.get("is_correct")
        annotations = record.get("principlism")
        if not isinstance(response, str) or status not in {"answered", "refusal", "unparseable"}:
            raise ValueError("Each result must have original_response and a valid response_status")
        if status == "answered":
            if model_answer not in VALID_ANSWERS or not isinstance(is_correct, bool):
                raise ValueError("Answered records need a valid model_answer and boolean is_correct")
            counts["answered"] += 1
            counts["correct" if is_correct else "incorrect"] += 1
        elif model_answer is not None or is_correct is not None:
            raise ValueError("Refusal and unparseable records must use null model_answer and is_correct")
        else:
            counts[status] += 1
        if isinstance(record.get("model"), str):
            models.add(record["model"])
        if not isinstance(annotations, dict) or not all(isinstance(value, bool) for value in annotations.values()):
            raise ValueError("Each result must include boolean principlism annotations")
        for principle, applies in annotations.items():
            if not applies:
                continue
            principle_counts = principles.setdefault(principle, {"total": 0, "answered": 0, "correct": 0})
            principle_counts["total"] += 1
            if status == "answered":
                principle_counts["answered"] += 1
                if is_correct:
                    principle_counts["correct"] += 1

    total = len(materialized_records)
    by_principle = {
        name: {
            **values,
            "accuracy_answered": values["correct"] / values["answered"] if values["answered"] else None,
            "accuracy_all": values["correct"] / values["total"] if values["total"] else None,
        }
        for name, values in sorted(principles.items())
    }
    return {
        "total": total,
        "models": sorted(models),
        **counts,
        "answer_rate": counts["answered"] / total if total else None,
        "accuracy_answered": counts["correct"] / counts["answered"] if counts["answered"] else None,
        "accuracy_all": counts["correct"] / total if total else None,
        "refusal_rate": counts["refusal"] / total if total else None,
        "unparseable_rate": counts["unparseable"] / total if total else None,
        "by_principle": by_principle,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(summary, target, ensure_ascii=False, indent=2)
            target.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="MCQA dataset JSON path")
    parser.add_argument("--output", type=Path, help="Evaluation output JSON path; required for evaluation")
    parser.add_argument("--analyze", type=Path, help="Analyze an existing result JSON instead of evaluating")
    parser.add_argument("--summary-output", type=Path, help="Optional path to write the JSON summary")
    parser.add_argument("--model", help="OpenAI-compatible model name")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key-env",
        default="MCQA_API_KEY",
        help="Environment variable holding the API key (default: MCQA_API_KEY)",
    )
    parser.add_argument("--question-id", action="append", type=int, help="Evaluate a question_id; repeat for multiple questions")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N selected questions")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent item workers (default: 4)")
    parser.add_argument("--retries", type=int, default=4, help="API attempts per item (default: 4)")
    parser.add_argument("--save-interval", type=int, default=10, help="Checkpoint after this many completed items (default: 10)")
    parser.add_argument("--force", action="store_true", help="Regenerate completed compatible records")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print selected questions without API calls")
    args = parser.parse_args()
    if args.analyze and (args.output or args.model or args.dry_run):
        parser.error("--analyze cannot be combined with evaluation options")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.retries < 1 or args.save_interval < 1:
        parser.error("--workers, --retries, and --save-interval must be positive")
    if not args.analyze and not args.dry_run:
        if args.output is None:
            parser.error("--output is required for evaluation")
        if not args.model:
            parser.error("--model is required for evaluation")
    return args


def main() -> int:
    args = parse_arguments()
    if args.analyze:
        try:
            summary = summarize_records(load_output(args.analyze).values())
        except ValueError as error:
            print(f"Analysis failed: {error}", file=sys.stderr)
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.summary_output:
            write_summary(args.summary_output, summary)
        return 0

    try:
        items = load_items(args.data)
    except ValueError as error:
        print(f"Dataset validation failed: {error}", file=sys.stderr)
        return 2
    selected_ids = set(args.question_id) if args.question_id else None
    selected = [item for item in items if selected_ids is None or item.question_id in selected_ids]
    missing_ids = selected_ids - {item.question_id for item in selected} if selected_ids else set()
    if missing_ids:
        print(f"Unknown question_ids: {sorted(missing_ids)}", file=sys.stderr)
        return 2
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        print("No questions selected.", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Validated {len(items)} MCQA items; selected {len(selected)} item(s).")
        for item in selected:
            print(f"question_id={item.question_id} id={item.item_id} correct_answer={item.correct_answer} question={item.question}")
        return 0

    assert args.output is not None and args.model is not None
    config = EndpointConfig(model=args.model, api_key_env=args.api_key_env, base_url=args.base_url)
    try:
        existing_records = load_output(args.output)
        client = get_client(config)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    records = dict(existing_records)
    completed = 0
    failures: list[str] = []
    lock = Lock()
    items_to_call = [
        item
        for item in selected
        if args.force or not record_is_reusable(existing_records.get(item.question_id), item, config.model)
    ]
    print(f"Selected {len(selected)} item(s); {len(items_to_call)} require model calls. Using {args.workers} worker(s).")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[dict[str, Any]], McqaItem] = {
            executor.submit(evaluate_item, item, existing_records.get(item.question_id), client, config, args.retries): item
            for item in items_to_call
        }
        with tqdm(total=len(futures), desc="Evaluating MCQA", unit="question") as progress:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    message = f"question_id {item.question_id} failed: {error}"
                    failures.append(message)
                    tqdm.write(message, file=sys.stderr)
                else:
                    with lock:
                        records[item.question_id] = record
                        completed += 1
                        if completed % args.save_interval == 0:
                            write_output(args.output, records.values(), config)
                            tqdm.write(f"Checkpointed {completed}/{len(items_to_call)} completed item(s).")
                finally:
                    progress.update(1)

    write_output(args.output, records.values(), config)
    completed_summary = summarize_records(records.values())
    print(json.dumps(completed_summary, ensure_ascii=False, indent=2))
    if args.summary_output:
        write_summary(args.summary_output, completed_summary)
    if failures:
        print(f"Completed with {len(failures)} failed item(s); rerun the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
