from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.agent.planner import Intent, build_planner
from rag.experiments.cost import PRICING
from rag.experiments.dataset import Query, load_queries

load_dotenv()

TYPE_TO_INTENT = {
    "single_program": Intent.single_program_details.value,
    "comparison": Intent.compare_programs.value,
    "aggregation": Intent.analytical_query.value,
    "multi_hop": Intent.analytical_query.value,
    "knowledge_lookup": Intent.knowledge_lookup.value,
    "refusal": Intent.off_topic_refuse.value,
}

DEFAULT_MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v4-flash",
]


def expected_intent(query: Query) -> str:
    if query.expected_refusal:
        return Intent.off_topic_refuse.value

    text = query.text.lower()
    if "дедлайн" in text or "срок" in text or "когда подача" in text:
        return Intent.admission_deadlines.value

    return TYPE_TO_INTENT[query.query_type]


def empty_confusion() -> dict[str, dict[str, int]]:
    intents = [intent.value for intent in Intent]
    return {expected: dict.fromkeys(intents, 0) for expected in intents}


def build_payload(rows: list[dict[str, Any]], with_confusion: bool = True) -> dict[str, Any]:
    correct = sum(1 for row in rows if row["expected_intent"] == row["predicted_intent"])
    error_count = sum(1 for row in rows if row["predicted_intent"] == "ERROR")
    by_type: dict[str, dict[str, float | int]] = {}
    confusion = empty_confusion()
    valid_intents = {intent.value for intent in Intent}

    for row in rows:
        query_type = row["query_type"]
        by_type.setdefault(query_type, {"correct": 0, "total": 0, "accuracy": 0.0})
        by_type[query_type]["total"] += 1
        if row["expected_intent"] == row["predicted_intent"]:
            by_type[query_type]["correct"] += 1
        predicted = row["predicted_intent"]
        if predicted in valid_intents:
            confusion[row["expected_intent"]][predicted] += 1

    for item in by_type.values():
        item["accuracy"] = item["correct"] / item["total"] if item["total"] else 0.0

    payload: dict[str, Any] = {
        "overall_accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "errors": error_count,
        "by_query_type": by_type,
    }
    if with_confusion:
        payload["confusion_matrix"] = confusion
    return payload


def _cache_path_for(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return Path("experiments/cache") / f"planner_{safe}.json"


async def run_for_model(
    queries: list[Query],
    prompt_path: Path,
    model_id: str,
) -> tuple[list[dict[str, Any]], list[float], int, int, int]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_in = 0
    total_out = 0
    n_with_tokens = 0
    os.environ["PLANNER_MODEL"] = model_id

    cache_path = _cache_path_for(model_id)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    agent = build_planner(prompt_path)

    for query in queries:
        cached = cache.get(query.query_id)
        if cached is not None and "input_tokens" in cached:
            predicted = cached["predicted_intent"]
            focus = cached.get("focus", "")
            latency_ms = cached.get("latency_ms", 0.0)
            in_tok = cached.get("input_tokens", 0)
            out_tok = cached.get("output_tokens", 0)
        else:
            t0 = time.perf_counter()
            in_tok = 0
            out_tok = 0
            try:
                result = await agent.run(query.text)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                plan = result.output
                predicted = plan.steps[0].intent.value if plan.steps else "ERROR"
                focus = plan.steps[0].focus if plan.steps else ""
                usage = result.usage()
                in_tok = getattr(usage, "input_tokens", 0) or 0
                out_tok = getattr(usage, "output_tokens", 0) or 0
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                predicted = "ERROR"
                focus = f"exception: {type(e).__name__}: {str(e)[:200]}"
            cache[query.query_id] = {
                "predicted_intent": predicted,
                "focus": focus,
                "latency_ms": latency_ms,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

        latencies.append(latency_ms)
        if in_tok or out_tok:
            total_in += in_tok
            total_out += out_tok
            n_with_tokens += 1
        expected = expected_intent(query)
        rows.append(
            {
                "qid": query.query_id,
                "query": query.text,
                "query_type": query.query_type,
                "expected_intent": expected,
                "predicted_intent": predicted,
                "focus": focus,
                "correct": expected == predicted,
            }
        )
    return rows, latencies, total_in, total_out, n_with_tokens


def latency_p50(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    sorted_lat = sorted(latencies)
    return sorted_lat[len(sorted_lat) // 2]


async def run(
    queries_path: Path,
    output_path: Path,
    prompt_path: Path,
    models: list[str],
) -> None:
    queries = load_queries(queries_path)
    by_model: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []

    for idx, model_id in enumerate(models):
        print(f"[{idx + 1}/{len(models)}] running planner with model: {model_id}")
        rows, latencies, total_in, total_out, n_tok = await run_for_model(
            queries, prompt_path, model_id
        )
        with_confusion = idx == 0
        payload_for_model = build_payload(rows, with_confusion=with_confusion)
        payload_for_model["latency_p50_ms"] = latency_p50(latencies)
        if n_tok and model_id in PRICING:
            in_price, out_price = PRICING[model_id]
            cost = (total_in * in_price + total_out * out_price) / 1_000_000
            payload_for_model["cost_per_100q_usd"] = cost * 100 / n_tok
            payload_for_model["avg_input_tokens"] = total_in / n_tok
            payload_for_model["avg_output_tokens"] = total_out / n_tok
            payload_for_model["n_token_samples"] = n_tok
        by_model[model_id] = payload_for_model
        for row in rows:
            row["model"] = model_id
            all_rows.append(row)

    baseline_model = models[0]
    final = {
        "family": "plan_accuracy",
        "config": {
            "n_queries": len(queries),
            "expected_mapping": TYPE_TO_INTENT,
            "models": models,
            "baseline_model": baseline_model,
        },
        "metrics": by_model[baseline_model],
        "by_model": by_model,
        "per_query": all_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Written: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("experiments/golden_set/validation.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/validation/plan_accuracy.json"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("rag/agent/prompts/planner.yaml"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )
    args = parser.parse_args()

    asyncio.run(run(args.queries, args.output, args.prompt, args.models))


if __name__ == "__main__":
    main()
