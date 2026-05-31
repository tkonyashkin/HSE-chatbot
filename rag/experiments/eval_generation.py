from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRow:
    query_id: str
    query_type: str
    is_refusal_query: bool
    answer: str


@dataclass(frozen=True)
class GenerationVariantResult:
    variant: str
    rows: list[GenerationRow]
    n_info: int
    n_refusal: int


def aggregate_generation(variant: str, rows: list[GenerationRow]) -> GenerationVariantResult:
    return GenerationVariantResult(
        variant=variant,
        rows=rows,
        n_info=sum(1 for r in rows if not r.is_refusal_query),
        n_refusal=sum(1 for r in rows if r.is_refusal_query),
    )


def compute_metrics(
    *,
    faithfulness_scores: dict[str, float],
    correctness_scores: dict[str, float],
    refusal_scores: dict[str, float],
    info_qids: list[str],
    refusal_qids: list[str],
    info_refusal_lookup: dict[str, bool],
    refusal_score_threshold: float = 0.5,
) -> dict[str, float]:
    n_info = len(info_qids)
    n_refusal = len(refusal_qids)

    tp = sum(1 for q in refusal_qids if refusal_scores.get(q, 0.0) >= refusal_score_threshold)
    fn = n_refusal - tp
    fp = sum(1 for q in info_qids if info_refusal_lookup.get(q, False))
    tn = n_info - fp

    answered_info_qids = [q for q in info_qids if not info_refusal_lookup.get(q, False)]
    n_answered = len(answered_info_qids)

    correctness_on_answered = (
        sum(correctness_scores.get(q, 0.0) for q in answered_info_qids) / n_answered
        if n_answered > 0
        else 0.0
    )
    answer_rate = n_answered / n_info if n_info > 0 else 0.0

    total = tp + tn + fp + fn
    refusal_accuracy_symmetric = (tp + tn) / total if total > 0 else 0.0
    false_refusal_rate = fp / n_info if n_info > 0 else 0.0

    faithfulness_mean = (
        sum(faithfulness_scores.values()) / len(faithfulness_scores) if faithfulness_scores else 0.0
    )

    return {
        "faithfulness": faithfulness_mean,
        "correctness_on_answered": correctness_on_answered,
        "refusal_accuracy_symmetric": refusal_accuracy_symmetric,
        "false_refusal_rate": false_refusal_rate,
        "answer_rate": answer_rate,
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "tn": float(tn),
        "n_info": float(n_info),
        "n_refusal": float(n_refusal),
        "n_answered": float(n_answered),
    }
