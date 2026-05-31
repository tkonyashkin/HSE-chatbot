from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.schemas import RetrievalHit

VANILLA_TEXT_PROMPT = (
    "Ты — консультант приёмной комиссии НИУ ВШЭ. Ответь на вопрос абитуриента "
    "СТРОГО на основе фрагментов с сайта ВШЭ ниже. Не выдумывай. Если данных "
    "в фрагментах недостаточно — скажи 'данных нет, обратитесь в приёмную "
    "комиссию ВШЭ'. Ответь кратко, по делу, на русском.\n\n"
    "Фрагменты:\n{context}\n\n"
    "Вопрос: {query}\n\n"
    "Ответ:"
)


class CachedAnswerer:
    def __init__(
        self,
        *,
        cache_path: Path,
        generate_fn: Callable[[str, list[RetrievalHit]], str],
    ) -> None:
        self._cache_path = cache_path
        self._generate = generate_fn
        self._cache: dict[str, str] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def __call__(self, query: str, hits: list[RetrievalHit]) -> str:
        if query in self._cache:
            return self._cache[query]
        answer = self._generate(query, hits)
        self._cache[query] = answer
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return answer


def build_vanilla_text(
    cache_dir: Path = Path("experiments/cache"),
    top_k_for_prompt: int = 5,
) -> CachedAnswerer:
    from rag.experiments.llm_preprocessors import default_llm_call

    llm = default_llm_call()

    def generate(query: str, hits: list[RetrievalHit]) -> str:
        top = hits[:top_k_for_prompt]
        context = "\n\n---\n\n".join(
            f"[Источник {i + 1}: {h.chunk.program_slug}/{h.chunk.chunk_type.value}]\n{h.chunk.text}"
            for i, h in enumerate(top)
        )
        prompt = VANILLA_TEXT_PROMPT.format(context=context, query=query)
        return llm(prompt).strip()

    return CachedAnswerer(
        cache_path=cache_dir / "vanilla_text.json",
        generate_fn=generate,
    )


def build_agentic_rag(
    cache_dir: Path = Path("experiments/cache"),
    *,
    agent_context: Any,
    prompts_dir: Path = Path("rag/agent/prompts"),
) -> CachedAnswerer:
    from rag.agent.runner import run_agent_planned_sync

    def generate(query: str, _hits: list[RetrievalHit]) -> str:
        response = run_agent_planned_sync(
            query=query, context=agent_context, worker_prompts_dir=prompts_dir
        )
        return response.text.strip()

    return CachedAnswerer(
        cache_path=cache_dir / "agentic_rag.json",
        generate_fn=generate,
    )
