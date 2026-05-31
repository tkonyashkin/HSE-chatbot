from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.experiments.cost import cost_usd


@dataclass
class CallRecord:
    model_id: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost: float


@dataclass
class CallTracker:
    model_id: str
    records: list[CallRecord] = field(default_factory=list)

    def record(self, *, latency_ms: float, prompt_tokens: int, completion_tokens: int) -> None:
        self.records.append(
            CallRecord(
                model_id=self.model_id,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost_usd(self.model_id, prompt_tokens, completion_tokens),
            )
        )

    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    def latency_percentile(self, p: float) -> float:
        if not self.records:
            return 0.0
        sorted_lat = sorted(r.latency_ms for r in self.records)
        idx = int(p * (len(sorted_lat) - 1))
        return sorted_lat[idx]


def now_ms() -> float:
    return time.perf_counter() * 1000.0
