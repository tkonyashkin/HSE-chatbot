from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.experiments.llm_preprocessors import load_prompt_template
from rag.experiments.runner import RetrievalPipeline
from rag.schemas import RetrievalHit

Reranker = Callable[[str, list[RetrievalHit]], list[RetrievalHit]]

LLM_RERANK_PROMPT = load_prompt_template(Path("rag/experiments/prompts/rerank.yaml"))


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def __call__(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return hits
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self._model.predict(pairs)
        scored = list(zip(scores, hits, strict=True))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return [h for _, h in scored]


class LLMReranker:
    def __init__(
        self,
        *,
        llm_call: Callable[[str], str],
        cache_path: Path,
        text_chars_per_candidate: int = 400,
    ) -> None:
        self._llm = llm_call
        self._cache_path = cache_path
        self._text_chars = text_chars_per_candidate
        self._cache: dict[str, list[int]] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def __call__(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return hits
        key = f"{query}|{','.join(h.chunk.chunk_id for h in hits)}"
        order: list[int]
        if key in self._cache:
            order = self._cache[key]
        else:
            try:
                order = self._call_llm(query, hits)
            except Exception:
                order = list(range(len(hits)))
            self._cache[key] = order
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return [hits[i] for i in order]

    def _call_llm(self, query: str, hits: list[RetrievalHit]) -> list[int]:
        candidates = "\n".join(
            f"[{i + 1}] {h.chunk.text[: self._text_chars].strip()}" for i, h in enumerate(hits)
        )
        prompt = LLM_RERANK_PROMPT.format(n=len(hits), query=query, candidates=candidates)
        response = self._llm(prompt).strip()
        return parse_ranking(response, len(hits))


def parse_ranking(response: str, n: int) -> list[int]:
    parsed: list[int] = []
    for token in response.replace("\n", ",").split(","):
        token = token.strip().lstrip("[(").rstrip("]).,")
        if not token.isdigit():
            continue
        idx = int(token) - 1
        if 0 <= idx < n and idx not in parsed:
            parsed.append(idx)
    for i in range(n):
        if i not in parsed:
            parsed.append(i)
    return parsed


class RerankingPipeline:
    def __init__(
        self,
        *,
        base: RetrievalPipeline,
        reranker: Reranker,
        pool_size: int = 20,
    ) -> None:
        self._base = base
        self._reranker = reranker
        self._pool_size = pool_size

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        pool = self._base.search(query, top_k=max(self._pool_size, top_k))
        reordered = self._reranker(query, pool)
        return reordered[:top_k]


def build_cross_encoder_reranker() -> Any:
    return CrossEncoderReranker()


def build_llm_reranker(
    cache_dir: Path = Path("experiments/cache"),
    model_id: str | None = None,
    tracker: Any = None,
    extra_body: dict[str, object] | None = None,
) -> Any:
    from rag.experiments.llm_preprocessors import default_llm_call

    suffix = ""
    if model_id:
        suffix = "_" + model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return LLMReranker(
        llm_call=default_llm_call(model_id=model_id, tracker=tracker, extra_body=extra_body),
        cache_path=cache_dir / f"llm_reranker{suffix}.json",
    )
