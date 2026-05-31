from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import TypeAdapter

from parser.knowledge_schemas import KnowledgeDoc
from parser.schemas import DeadlineEntry
from rag.agent.context import AgentContext
from rag.agent.runner import run_agent_planned_sync
from rag.agent.tools import derive_acronyms_from_qdrant
from rag.embedding.embedder import Embedder, EmbedderConfig
from rag.experiments.dataset import load_queries
from rag.experiments.deepeval_judge import build_judge
from rag.experiments.eval_generation import compute_metrics
from rag.experiments.preprocessors import expand_acronyms
from rag.experiments.rerankers import RerankingPipeline, build_llm_reranker
from rag.indexing.indexer import IndexerConfig
from rag.indexing.indexer import _client as _qdrant_client
from rag.retrieval.pipeline import RetrievalPipeline

load_dotenv()

DEFAULT_MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v4-flash",
]

PLANNER_MODEL = "anthropic/claude-sonnet-4.6"
RERANKER_MODEL = "deepseek/deepseek-v4-flash"
RERANKER_EXTRA_BODY: dict[str, object] = {"reasoning": {"enabled": False}}

_KNOWLEDGE_DOC_ADAPTER: TypeAdapter[KnowledgeDoc] = TypeAdapter(KnowledgeDoc)


def _load_deadlines(knowledge_dir: Path) -> list[DeadlineEntry]:
    out: list[DeadlineEntry] = []
    if not knowledge_dir.exists():
        return out
    for json_file in sorted(knowledge_dir.glob("*.json")):
        doc = _KNOWLEDGE_DOC_ADAPTER.validate_json(json_file.read_text(encoding="utf-8"))
        out.extend(getattr(doc, "deadlines", []))
    return out


def _cache_path_for(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace("-", "_").replace(".", "_").replace("#", "_")
    return Path("experiments/cache") / f"executor_{safe}.json"


REASONING_ON_SUFFIX = "#reasoning_on"


def _strip_reasoning_suffix(model_id: str) -> tuple[str, bool]:
    if model_id.endswith(REASONING_ON_SUFFIX):
        return model_id[: -len(REASONING_ON_SUFFIX)], True
    return model_id, False


def build_context(
    embedding_config_path: Path,
    retrieval_config_path: Path,
    acronyms_path: Path,
    knowledge_dir: Path,
) -> tuple[AgentContext, RerankingPipeline]:
    emb_config = EmbedderConfig.from_yaml(embedding_config_path)
    embedder = Embedder(emb_config)
    index_config = IndexerConfig.from_yaml(retrieval_config_path, dim=emb_config.dim)
    retrieval = RetrievalPipeline(embedder=embedder, index_config=index_config)
    qc = _qdrant_client(index_config.qdrant_path)
    deadlines = _load_deadlines(knowledge_dir)
    reranker_pipe = RerankingPipeline(
        base=retrieval,
        reranker=build_llm_reranker(model_id=RERANKER_MODEL, extra_body=RERANKER_EXTRA_BODY),
        pool_size=20,
    )
    context = AgentContext(
        retrieval=reranker_pipe,
        qdrant_client=qc,
        qdrant_collection=index_config.collection_name,
        acronyms=derive_acronyms_from_qdrant(qc, index_config.collection_name, acronyms_path),
        deadlines=deadlines,
    )
    return context, reranker_pipe


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(p * (len(sorted_v) - 1))
    return sorted_v[idx]


def run_for_model(
    queries: list[Any],
    context: AgentContext,
    model_id: str,
) -> dict[str, dict[str, Any]]:
    cache_path = _cache_path_for(model_id)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    actual_model, reasoning_on = _strip_reasoning_suffix(model_id)
    os.environ["OPENROUTER_MODEL"] = actual_model
    os.environ["PLANNER_MODEL"] = PLANNER_MODEL
    if actual_model.startswith("deepseek/") and not reasoning_on:
        os.environ["EXECUTOR_EXTRA_BODY"] = json.dumps({"reasoning": {"enabled": False}})
    else:
        os.environ.pop("EXECUTOR_EXTRA_BODY", None)

    for i, q in enumerate(queries):
        if q.query_id in cache:
            continue
        t0 = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        requests = 0
        try:
            response = run_agent_planned_sync(query=q.text, context=context)
            answer = response.text
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            requests = response.requests
            error = None
        except Exception as e:
            answer = ""
            error = f"{type(e).__name__}: {str(e)[:200]}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        cache[q.query_id] = {
            "answer": answer,
            "latency_ms": latency_ms,
            "error": error,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests": requests,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        if (i + 1) % 10 == 0:
            print(f"  [{model_id}] {i + 1}/{len(queries)}")

    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries", type=Path, default=Path("experiments/golden_set/validation.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/validation/executor_model.json"),
    )
    parser.add_argument("--embedding-config", type=Path, default=Path("config/embedding.yaml"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("config/retrieval.yaml"))
    parser.add_argument("--acronyms", type=Path, default=Path("config/acronyms.yaml"))
    parser.add_argument("--knowledge-dir", type=Path, default=Path("data/knowledge/2026"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    info_qids = [q.query_id for q in queries if not q.expected_refusal]
    refusal_qids = [q.query_id for q in queries if q.expected_refusal]

    print(f"Loading context (planner={PLANNER_MODEL}, reranker={RERANKER_MODEL})…")
    context, reranker_pipe = build_context(
        args.embedding_config, args.retrieval_config, args.acronyms, args.knowledge_dir
    )

    print("Loading judge (Haiku 4.5)…")
    judge = build_judge()

    by_model: dict[str, dict[str, Any]] = {}
    per_query_all: list[dict[str, Any]] = []

    for idx, model_id in enumerate(args.models):
        print(f"\n[{idx + 1}/{len(args.models)}] running executor with model: {model_id}")
        answers = run_for_model(queries, context, model_id)

        print(f"  judging {model_id} answers…")
        faithfulness: dict[str, float] = {}
        correctness: dict[str, float] = {}
        refusal_scores: dict[str, float] = {}
        info_refusal_lookup: dict[str, bool] = {}
        for q in queries:
            entry = answers[q.query_id]
            ans = entry["answer"]
            if entry.get("error"):
                if not q.expected_refusal:
                    info_refusal_lookup[q.query_id] = True
                continue
            if q.expected_refusal:
                try:
                    refusal_scores[q.query_id] = judge.score_refusal(
                        variant=model_id, query_id=q.query_id, query=q.text, answer=ans
                    )
                except Exception as exc:
                    print(
                        f"  judge.score_refusal failed on {q.query_id}: {type(exc).__name__}; skip"
                    )
            else:
                pre = expand_acronyms(q.text, context.acronyms)
                hits = reranker_pipe.search(pre, top_k=5)
                try:
                    score = judge.score(
                        variant=model_id,
                        query_id=q.query_id,
                        query=q.text,
                        expected_facts=list(q.expected_answer_facts),
                        retrieved=hits,
                        answer=ans,
                    )
                    faithfulness[q.query_id] = score.faithfulness
                    correctness[q.query_id] = score.correctness
                except Exception as exc:
                    print(f"  judge.score failed on {q.query_id}: {type(exc).__name__}; skip")
                try:
                    info_refusal_lookup[q.query_id] = judge.is_refusal_on_info(
                        variant=model_id, query_id=q.query_id, query=q.text, answer=ans
                    )
                except Exception as exc:
                    print(
                        f"  judge.is_refusal_on_info failed on {q.query_id}: "
                        f"{type(exc).__name__}; skip"
                    )
                    info_refusal_lookup[q.query_id] = False

        m = compute_metrics(
            faithfulness_scores=faithfulness,
            correctness_scores=correctness,
            refusal_scores=refusal_scores,
            info_qids=info_qids,
            refusal_qids=refusal_qids,
            info_refusal_lookup=info_refusal_lookup,
        )
        latencies = [e["latency_ms"] for e in answers.values() if not e.get("error")]
        error_count = sum(1 for e in answers.values() if e.get("error"))
        m["latency_p50_ms"] = percentile(latencies, 0.5)
        m["latency_p95_ms"] = percentile(latencies, 0.95)
        m["errors"] = float(error_count)

        from rag.experiments.cost import PRICING

        tracked = [
            e for e in answers.values() if not e.get("error") and e.get("input_tokens", 0) > 0
        ]
        n_tracked = len(tracked)
        if n_tracked:
            total_input = sum(e["input_tokens"] for e in tracked)
            total_output = sum(e["output_tokens"] for e in tracked)
            m["avg_input_tokens"] = total_input / n_tracked
            m["avg_output_tokens"] = total_output / n_tracked
            m["avg_requests"] = sum(e.get("requests", 0) for e in tracked) / n_tracked
            m["n_cost_samples"] = float(n_tracked)
            actual_for_pricing, _ = _strip_reasoning_suffix(model_id)
            if actual_for_pricing in PRICING:
                in_price, out_price = PRICING[actual_for_pricing]
                m["cost_per_100q_usd"] = (
                    (total_input * in_price + total_output * out_price)
                    / 1_000_000
                    * 100
                    / n_tracked
                )
        by_model[model_id] = m

        for q in queries:
            e = answers[q.query_id]
            per_query_all.append(
                {
                    "qid": q.query_id,
                    "query_type": q.query_type,
                    "is_refusal_query": q.expected_refusal,
                    "model": model_id,
                    "answer": e["answer"],
                    "latency_ms": e["latency_ms"],
                    "error": e.get("error"),
                    "faithfulness": faithfulness.get(q.query_id),
                    "correctness": correctness.get(q.query_id),
                    "refusal_score": refusal_scores.get(q.query_id),
                }
            )

    payload = {
        "family": "executor_model",
        "config": {
            "queries_path": str(args.queries),
            "models": args.models,
            "n_queries": len(queries),
            "planner_model": PLANNER_MODEL,
            "reranker_model": RERANKER_MODEL,
            "judge_model": "anthropic/claude-haiku-4.5",
            "retrieval": "glossary + dense e5_large + llm_haiku reranker (FIXED)",
            "composite": "correctness_on_answered × refusal_accuracy_symmetric",
        },
        "metrics": by_model,
        "per_query": per_query_all,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
