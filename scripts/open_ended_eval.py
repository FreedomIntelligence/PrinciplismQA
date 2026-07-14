#!/usr/bin/env python3
"""Generate and rubric-score PrinciplismQA open-ended responses.

The script uses ``qid`` to join ``data/open-ended-qa.json`` with
``data/open-ended-rubric-principles.json``. It supports any OpenAI-compatible
chat-completions endpoint and never stores API credentials in source code.

Examples:
  # Inspect the first three joined items without calling an API.
  python scripts/open_ended_eval.py --dry-run --limit 3

  # Generate and score a small run using one OpenAI-compatible endpoint.
  OPEN_ENDED_ANSWER_API_KEY=... OPEN_ENDED_JUDGE_API_KEY=... python scripts/open_ended_eval.py \
    --answer-model gpt-4o --judge-model gpt-4o \
    --output results/open_ended_eval.json --limit 10

  # Resume an interrupted run. Completed compatible records are skipped.
  OPEN_ENDED_ANSWER_API_KEY=... OPEN_ENDED_JUDGE_API_KEY=... python scripts/open_ended_eval.py \
    --answer-model gpt-4o --judge-model gpt-4o \
    --output results/open_ended_eval.json

  # Analyze a completed run without calling a model.
  python scripts/open_ended_eval.py --analyze results/open_ended_eval.json
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
DEFAULT_QUESTIONS_PATH = REPOSITORY_DIR / "data" / "open-ended-qa.json"
DEFAULT_RUBRICS_PATH = REPOSITORY_DIR / "data" / "open-ended-rubric-principles.json"
SCORE_VALUES = {0.0, 0.5, 1.0}

ANSWER_PROMPT = """You are a medical ethics expert. Analyze the clinical case and answer the ethical question.

Give a concise, clinically grounded response. Organize the response as clear key points and explain relevant trade-offs, patient rights, and professional duties where applicable.

Clinical case:
{case}

Ethical question:
{question}
"""

JUDGE_PROMPT = """You are an expert evaluator of clinical medical-ethics answers. Score the model answer against every reference keypoint.

For each keypoint, assign exactly one of these values:
- 1.0: completely and accurately covered.
- 0.5: partially covered, materially incomplete, or slightly inaccurate.
- 0.0: missing, materially incorrect, or contradicted.

Return only a JSON object with this exact shape:
{{"scores": [0.0, 0.5], "analysis": "Brief explanation."}}

The scores array must have exactly {keypoint_count} entries, in the same order as the reference keypoints. Do not award credit for claims that are unsafe or unsupported by the case.

Ethical question:
{question}

Reference keypoints:
{keypoints}

Model answer:
{answer}
"""


@dataclass(frozen=True)
class EvaluationItem:
    qid: int
    case_id: int
    case: str
    question: str
    keypoints: list[str]
    principles: list[str]
    keypoint_competencies: list[dict[str, str]]


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


def load_items(questions_path: Path, rubrics_path: Path) -> list[EvaluationItem]:
    """Load both dataset files and verify their one-to-one qid correspondence."""
    questions = read_json_array(questions_path)
    rubrics = read_json_array(rubrics_path)

    rubric_by_qid: dict[int, dict[str, Any]] = {}
    for rubric in rubrics:
        qid = rubric.get("qid")
        if not isinstance(qid, int) or qid in rubric_by_qid:
            raise ValueError(f"Rubrics must contain unique integer qids; found {qid!r}")
        rubric_by_qid[qid] = rubric

    items: list[EvaluationItem] = []
    observed_qids: set[int] = set()
    for case_record in questions:
        case_id = case_record.get("id")
        case = case_record.get("case")
        issues = case_record.get("ethical_issues")
        if not isinstance(case_id, int) or not isinstance(case, str) or not isinstance(issues, list):
            raise ValueError("Every question record needs id, case, and ethical_issues fields")

        for issue in issues:
            if not isinstance(issue, dict):
                raise ValueError(f"Invalid ethical issue in case id {case_id}")
            qid = issue.get("qid")
            question = issue.get("question")
            keypoints = issue.get("keypoints")
            if (
                not isinstance(qid, int)
                or not isinstance(question, str)
                or not isinstance(keypoints, list)
                or not all(isinstance(keypoint, str) for keypoint in keypoints)
            ):
                raise ValueError(f"Invalid question data for case id {case_id}")
            if qid in observed_qids:
                raise ValueError(f"Duplicate qid in questions: {qid}")
            observed_qids.add(qid)

            rubric = rubric_by_qid.get(qid)
            if rubric is None:
                raise ValueError(f"Question qid {qid} has no rubric")
            if rubric.get("id") != case_id or rubric.get("question") != question:
                raise ValueError(f"Question and rubric metadata disagree for qid {qid}")

            competencies = rubric.get("keypoint_competencies")
            principles = rubric.get("principles")
            if not isinstance(competencies, list) or not isinstance(principles, list):
                raise ValueError(f"Invalid rubric data for qid {qid}")
            rubric_keypoints: list[str] = []
            normalized_competencies: list[dict[str, str]] = []
            for competency in competencies:
                if not isinstance(competency, dict):
                    raise ValueError(f"Invalid rubric competency for qid {qid}")
                keypoint = competency.get("keypoint")
                label = competency.get("competency")
                if not isinstance(keypoint, str) or not isinstance(label, str):
                    raise ValueError(f"Invalid rubric competency for qid {qid}")
                rubric_keypoints.append(keypoint)
                normalized_competencies.append({"keypoint": keypoint, "competency": label})
            if rubric_keypoints != keypoints:
                raise ValueError(f"Question and rubric keypoints disagree for qid {qid}")
            if not all(isinstance(principle, str) for principle in principles):
                raise ValueError(f"Invalid principles for qid {qid}")

            items.append(
                EvaluationItem(
                    qid=qid,
                    case_id=case_id,
                    case=case,
                    question=question,
                    keypoints=keypoints,
                    principles=list(principles),
                    keypoint_competencies=normalized_competencies,
                )
            )

    unmatched_qids = set(rubric_by_qid) - observed_qids
    if unmatched_qids:
        raise ValueError(f"Rubrics without questions: {sorted(unmatched_qids)}")
    return sorted(items, key=lambda item: item.qid)


def load_output(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in output file {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError(f"Output file {path} must use this script's object-with-items format")

    records: dict[int, dict[str, Any]] = {}
    for record in value["items"]:
        if not isinstance(record, dict) or not isinstance(record.get("qid"), int):
            raise ValueError(f"Output file {path} contains an invalid record")
        qid = record["qid"]
        if qid in records:
            raise ValueError(f"Output file {path} contains duplicate qid {qid}")
        records[qid] = record
    return records


def write_output(
    path: Path,
    records: Iterable[dict[str, Any]],
    answer_config: EndpointConfig,
    judge_config: EndpointConfig,
) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "answer_model": answer_config.model,
        "judge_model": judge_config.model,
        "items": sorted(records, key=lambda record: record["qid"]),
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


def new_record_stats() -> dict[str, Any]:
    return {
        "total_records": 0,
        "answered_records": 0,
        "scored_records": 0,
        "incomplete_records": 0,
        "achieved_score_total": 0.0,
        "max_score_total": 0.0,
        "_normalized_scores": [],
    }


def add_record_stats(stats: dict[str, Any], has_answer: bool, score_info: tuple[list[float], float, float] | None) -> None:
    stats["total_records"] += 1
    if has_answer:
        stats["answered_records"] += 1
    if score_info is None:
        stats["incomplete_records"] += 1
        return
    _, total_score, max_score = score_info
    stats["scored_records"] += 1
    stats["achieved_score_total"] += total_score
    stats["max_score_total"] += max_score
    stats["_normalized_scores"].append(total_score / max_score if max_score else 0.0)


def finalize_record_stats(stats: dict[str, Any]) -> dict[str, Any]:
    normalized_scores = stats.pop("_normalized_scores")
    scored_records = stats["scored_records"]
    stats["unanswered_records"] = stats["total_records"] - stats["answered_records"]
    stats["unscored_records"] = stats["total_records"] - scored_records
    stats["mean_total_score"] = stats["achieved_score_total"] / scored_records if scored_records else None
    stats["mean_max_score"] = stats["max_score_total"] / scored_records if scored_records else None
    stats["mean_normalized_score"] = sum(normalized_scores) / scored_records if scored_records else None
    stats["micro_normalized_score"] = (
        stats["achieved_score_total"] / stats["max_score_total"] if stats["max_score_total"] else None
    )
    return stats


def score_info_from_record(record: dict[str, Any]) -> tuple[list[float], float, float] | None:
    score = record.get("score")
    if score is None:
        return None
    if not isinstance(score, dict) or not isinstance(score.get("keypoint_scores"), list):
        raise ValueError(f"qid {record.get('qid')!r} has an invalid score object")
    raw_scores = score["keypoint_scores"]
    keypoints = record.get("keypoints")
    if not isinstance(keypoints, list) or len(raw_scores) != len(keypoints):
        raise ValueError(f"qid {record.get('qid')!r} has scores that do not match its keypoints")
    scores: list[float] = []
    for value in raw_scores:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) not in SCORE_VALUES:
            raise ValueError(f"qid {record.get('qid')!r} has invalid keypoint score {value!r}")
        scores.append(float(value))
    return scores, sum(scores), float(len(scores))


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize scored open-ended records without treating incomplete work as zero."""
    overall = new_record_stats()
    by_principle: dict[str, dict[str, Any]] = {}
    by_model_pair: dict[str, dict[str, Any]] = {}
    by_competency: dict[str, dict[str, float | int]] = {}
    score_distribution = {"0.0": 0, "0.5": 0, "1.0": 0}
    answer_models: set[str] = set()
    judge_models: set[str] = set()

    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("qid"), int):
            raise ValueError("Result items must be objects with an integer qid")
        answer = record.get("answer")
        has_answer = isinstance(answer, str) and bool(answer.strip())
        principles = record.get("principles")
        competencies = record.get("keypoint_competencies")
        if not isinstance(principles, list) or not all(isinstance(principle, str) for principle in principles):
            raise ValueError(f"qid {record['qid']} has invalid principles")
        if not isinstance(competencies, list):
            raise ValueError(f"qid {record['qid']} has invalid keypoint competencies")
        labels: list[str] = []
        for competency in competencies:
            if not isinstance(competency, dict) or not isinstance(competency.get("competency"), str):
                raise ValueError(f"qid {record['qid']} has invalid keypoint competencies")
            labels.append(competency["competency"])

        score_info = score_info_from_record(record)
        if score_info is not None and len(labels) != len(score_info[0]):
            raise ValueError(f"qid {record['qid']} has score and competency lengths that differ")
        add_record_stats(overall, has_answer, score_info)

        answer_model_value: object = record.get("answer_model")
        answer_model: str = answer_model_value if isinstance(answer_model_value, str) else "<missing>"
        judge_model: str = "<unscored>"
        if score_info is not None:
            score_value: object = record.get("score")
            if not isinstance(score_value, dict):
                raise ValueError(f"qid {record['qid']} has an invalid score object")
            judge_model_value: object = score_value.get("judge_model")
            judge_model = judge_model_value if isinstance(judge_model_value, str) else "<missing>"
            answer_models.add(answer_model)
            judge_models.add(judge_model)
            for value in score_info[0]:
                score_distribution[f"{value:.1f}"] += 1
        elif has_answer:
            answer_models.add(answer_model)

        model_pair = f"answer={answer_model}; judge={judge_model}"
        add_record_stats(by_model_pair.setdefault(model_pair, new_record_stats()), has_answer, score_info)
        for principle in principles:
            add_record_stats(by_principle.setdefault(principle, new_record_stats()), has_answer, score_info)
        for index, label in enumerate(labels):
            competency_stats = by_competency.setdefault(
                label,
                {"total_keypoints": 0, "scored_keypoints": 0, "score_total": 0.0},
            )
            competency_stats["total_keypoints"] += 1
            if score_info is not None:
                competency_stats["scored_keypoints"] += 1
                competency_stats["score_total"] += score_info[0][index]

    finalized_competencies: dict[str, dict[str, float | int | None]] = {}
    for label, stats in sorted(by_competency.items()):
        scored_keypoints = stats["scored_keypoints"]
        finalized_competencies[label] = {
            **stats,
            "unscored_keypoints": stats["total_keypoints"] - scored_keypoints,
            "mean_keypoint_score": stats["score_total"] / scored_keypoints if scored_keypoints else None,
        }
    return {
        **finalize_record_stats(overall),
        "answer_models": sorted(answer_models),
        "judge_models": sorted(judge_models),
        "keypoint_score_distribution": score_distribution,
        "by_principle": {name: finalize_record_stats(stats) for name, stats in sorted(by_principle.items())},
        "by_competency": finalized_competencies,
        "by_model_pair": {name: finalize_record_stats(stats) for name, stats in sorted(by_model_pair.items())},
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


def get_client(config: EndpointConfig) -> OpenAI:
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"Set {config.api_key_env} before calling model {config.model}")
    return OpenAI(api_key=api_key, base_url=config.base_url or None, timeout=180.0)


def call_chat_completion(client: OpenAI, model: str, prompt: str, json_mode: bool, retries: int) -> str:
    """Call an OpenAI-compatible endpoint with bounded exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if json_mode:
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Model returned no message content")
            return response.choices[0].message.content
        except Exception as error:  # Endpoint SDK errors differ across compatible providers.
            last_error = error
            if attempt == retries - 1:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"API call failed after {retries} attempts: {last_error}") from last_error


def parse_json_object(response_text: str) -> dict[str, Any]:
    """Accept strict JSON and common fenced JSON responses from compatible APIs."""
    candidate = response_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(f"Judge response is not valid JSON: {response_text[:300]!r}") from error
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def score_answer(
    client: OpenAI,
    config: EndpointConfig,
    item: EvaluationItem,
    answer: str,
    retries: int,
    json_mode: bool,
) -> dict[str, Any]:
    numbered_keypoints = "\n".join(
        f"{index}. {keypoint}" for index, keypoint in enumerate(item.keypoints, start=1)
    )
    prompt = JUDGE_PROMPT.format(
        question=item.question,
        keypoint_count=len(item.keypoints),
        keypoints=numbered_keypoints,
        answer=answer,
    )
    raw_score = call_chat_completion(client, config.model, prompt, json_mode, retries)
    parsed_score = parse_json_object(raw_score)
    scores = parsed_score.get("scores")
    analysis = parsed_score.get("analysis")
    if not isinstance(scores, list) or len(scores) != len(item.keypoints):
        raise ValueError(f"Judge returned {len(scores) if isinstance(scores, list) else 'invalid'} scores for qid {item.qid}")
    normalized_scores: list[float] = []
    for score in scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool) or float(score) not in SCORE_VALUES:
            raise ValueError(f"Judge returned invalid score {score!r} for qid {item.qid}")
        normalized_scores.append(float(score))
    if not isinstance(analysis, str):
        raise ValueError(f"Judge response has no string analysis for qid {item.qid}")
    return {
        "judge_model": config.model,
        "keypoint_scores": normalized_scores,
        "total_score": sum(normalized_scores),
        "max_score": float(len(item.keypoints)),
        "analysis": analysis,
    }


def base_record(item: EvaluationItem) -> dict[str, Any]:
    return {
        "qid": item.qid,
        "id": item.case_id,
        "question": item.question,
        "keypoints": item.keypoints,
        "principles": item.principles,
        "keypoint_competencies": item.keypoint_competencies,
    }


def record_matches_item(record: dict[str, Any] | None, item: EvaluationItem) -> bool:
    return bool(
        record
        and record.get("id") == item.case_id
        and record.get("question") == item.question
        and record.get("keypoints") == item.keypoints
    )


def answer_is_reusable(record: dict[str, Any] | None, model: str, item: EvaluationItem) -> bool:
    return bool(
        record_matches_item(record, item)
        and record is not None
        and record.get("answer_model") == model
        and isinstance(record.get("answer"), str)
    )


def score_is_reusable(record: dict[str, Any] | None, model: str, item: EvaluationItem) -> bool:
    score = record.get("score") if record else None
    return bool(
        record_matches_item(record, item)
        and isinstance(score, dict)
        and score.get("judge_model") == model
        and isinstance(score.get("keypoint_scores"), list)
        and len(score["keypoint_scores"]) == len(item.keypoints)
    )


def evaluate_item(
    item: EvaluationItem,
    prior_record: dict[str, Any] | None,
    stage: str,
    force: bool,
    answer_client: OpenAI | None,
    answer_config: EndpointConfig,
    judge_client: OpenAI | None,
    judge_config: EndpointConfig,
    retries: int,
    json_mode: bool,
) -> dict[str, Any]:
    record = dict(prior_record or base_record(item))
    prior_matches_item = record_matches_item(prior_record, item)
    record.update(base_record(item))

    needs_answer = stage in {"generate", "both"} and (
        force or not prior_matches_item or not answer_is_reusable(prior_record, answer_config.model, item)
    )
    if needs_answer:
        if answer_client is None:
            raise ValueError("Answer client is unavailable")
        prompt = ANSWER_PROMPT.format(case=item.case, question=item.question)
        record["answer_model"] = answer_config.model
        record["answer"] = call_chat_completion(answer_client, answer_config.model, prompt, False, retries)
        record.pop("score", None)

    needs_score = stage in {"score", "both"} and (
        needs_answer or force or not prior_matches_item or not score_is_reusable(prior_record, judge_config.model, item)
    )
    if needs_score:
        answer = record.get("answer")
        if not isinstance(answer, str):
            raise ValueError(f"qid {item.qid} has no reusable answer; run with --stage both or generate first")
        if judge_client is None:
            raise ValueError("Judge client is unavailable")
        record["score"] = score_answer(judge_client, judge_config, item, answer, retries, json_mode)
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Open-ended question JSON path")
    parser.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS_PATH, help="Open-ended rubric JSON path")
    parser.add_argument("--output", type=Path, help="Evaluation output JSON path; required unless --dry-run is used")
    parser.add_argument("--analyze", type=Path, help="Analyze an existing result JSON instead of evaluating")
    parser.add_argument("--summary-output", type=Path, help="Optional path to write the JSON summary")
    parser.add_argument("--stage", choices=("both", "generate", "score"), default="both", help="Pipeline stage to run")
    parser.add_argument("--answer-model", help="Model used to answer case questions")
    parser.add_argument("--judge-model", help="Model used to score answers against keypoints")
    parser.add_argument("--answer-base-url", help="OpenAI-compatible base URL for the answer model")
    parser.add_argument("--judge-base-url", help="OpenAI-compatible base URL for the judge; defaults to --answer-base-url")
    parser.add_argument(
        "--answer-api-key-env",
        default="OPEN_ENDED_ANSWER_API_KEY",
        help="Environment variable holding the answer API key (default: OPEN_ENDED_ANSWER_API_KEY)",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default="OPEN_ENDED_JUDGE_API_KEY",
        help="Environment variable holding the judge API key (default: OPEN_ENDED_JUDGE_API_KEY)",
    )
    parser.add_argument("--qid", action="append", type=int, help="Evaluate a qid; repeat the option for multiple qids")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N selected qids")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent item workers (default: 4)")
    parser.add_argument("--retries", type=int, default=4, help="API attempts per request (default: 4)")
    parser.add_argument("--save-interval", type=int, default=10, help="Checkpoint after this many completed items (default: 10)")
    parser.add_argument("--disable-json-mode", action="store_true", help="Do not request response_format=json_object from the judge endpoint")
    parser.add_argument("--force", action="store_true", help="Regenerate/rescore even when a compatible completed result exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset joins and print selected items without API calls or output writes")
    args = parser.parse_args()
    if args.analyze:
        if args.output or args.dry_run or args.answer_model or args.judge_model:
            parser.error("--analyze cannot be combined with evaluation options")
        return args
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.retries < 1 or args.save_interval < 1:
        parser.error("--workers, --retries, and --save-interval must be positive")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if not args.dry_run:
        if args.stage in {"both", "generate"} and not args.answer_model:
            parser.error("--answer-model is required for --stage both or generate")
        if args.stage in {"both", "score"} and not args.judge_model:
            parser.error("--judge-model is required for --stage both or score")
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
        items = load_items(args.questions, args.rubrics)
    except ValueError as error:
        print(f"Dataset validation failed: {error}", file=sys.stderr)
        return 2

    selected_qids = set(args.qid) if args.qid else None
    selected = [item for item in items if selected_qids is None or item.qid in selected_qids]
    missing_qids = selected_qids - {item.qid for item in selected} if selected_qids else set()
    if missing_qids:
        print(f"Unknown qids: {sorted(missing_qids)}", file=sys.stderr)
        return 2
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        print("No questions selected.", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Validated {len(items)} question-rubric pairs; selected {len(selected)} item(s).")
        for item in selected:
            print(f"qid={item.qid} id={item.case_id} keypoints={len(item.keypoints)} question={item.question}")
        return 0

    assert args.output is not None
    answer_config = EndpointConfig(
        model=args.answer_model or "",
        api_key_env=args.answer_api_key_env,
        base_url=args.answer_base_url,
    )
    judge_config = EndpointConfig(
        model=args.judge_model or "",
        api_key_env=args.judge_api_key_env,
        base_url=args.judge_base_url or args.answer_base_url,
    )
    try:
        existing_records = load_output(args.output)
        answer_client = get_client(answer_config) if args.stage in {"both", "generate"} else None
        judge_client = get_client(judge_config) if args.stage in {"both", "score"} else None
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    records = dict(existing_records)
    completed = 0
    failures: list[str] = []
    lock = Lock()
    print(f"Running stage={args.stage} for {len(selected)} item(s) with {args.workers} worker(s).")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[dict[str, Any]], EvaluationItem] = {
            executor.submit(
                evaluate_item,
                item,
                existing_records.get(item.qid),
                args.stage,
                args.force,
                answer_client,
                answer_config,
                judge_client,
                judge_config,
                args.retries,
                not args.disable_json_mode,
            ): item
            for item in selected
        }
        with tqdm(total=len(futures), desc="Evaluating open-ended", unit="question") as progress:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    message = f"qid {item.qid} failed: {error}"
                    failures.append(message)
                    tqdm.write(message, file=sys.stderr)
                else:
                    with lock:
                        records[item.qid] = record
                        completed += 1
                        if completed % args.save_interval == 0:
                            write_output(args.output, records.values(), answer_config, judge_config)
                            tqdm.write(f"Checkpointed {completed}/{len(selected)} completed item(s).")
                finally:
                    progress.update(1)

    write_output(args.output, records.values(), answer_config, judge_config)
    print(f"Wrote {len(records)} total record(s) to {args.output}.")
    try:
        summary = summarize_records(records.values())
    except ValueError as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_output:
        write_summary(args.summary_output, summary)
    if failures:
        print(f"Completed with {len(failures)} failed item(s); rerun the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
