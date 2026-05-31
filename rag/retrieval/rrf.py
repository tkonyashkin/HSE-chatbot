from __future__ import annotations

from rag.schemas import RetrievalHit


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]],
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    candidates: dict[str, RetrievalHit] = {}

    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for hit in ranked:
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            candidates.setdefault(chunk_id, hit)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + hit.rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    fused: list[RetrievalHit] = []
    for rank, chunk_id in enumerate(ordered_ids[:top_k], start=1):
        fused.append(
            candidates[chunk_id].model_copy(update={"score": scores[chunk_id], "rank": rank})
        )
    return fused
