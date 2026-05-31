from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.experiments.runner import VariantResult


def write_json(
    path: Path,
    *,
    family: str,
    variants: list[VariantResult],
    config: dict[str, Any],
) -> None:
    metrics = {}
    for variant in variants:
        entry = {
            "Hit@5": variant.mean_hit_at_5,
            "MRR@5": variant.mean_mrr_at_5,
        }
        if family != "reranker":
            entry["Recall@20"] = variant.mean_recall_at_20
        metrics[variant.variant] = entry

    rows = []
    for variant in variants:
        for row in variant.rows:
            rows.append(
                {
                    "qid": row.query_id,
                    "query_type": row.query_type,
                    "variant": variant.variant,
                    "preprocessed_query": row.preprocessed_query,
                    "retrieved_keys": row.retrieved_keys,
                    "hit_at_5": bool(row.hit_at_5),
                    "mrr_at_5": row.mrr_at_5,
                    "recall_at_20": row.recall_at_20,
                }
            )

    payload = {
        "family": family,
        "config": {
            **config,
            "variants": [v.variant for v in variants],
            "n_queries": len(variants[0].rows) if variants else 0,
        },
        "metrics": metrics,
        "per_query": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
