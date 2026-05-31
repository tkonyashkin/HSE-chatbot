from __future__ import annotations

PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "deepseek/deepseek-v4-flash": (0.0983, 0.1966),
}


def cost_usd(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = PRICING[model_id]
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def cost_per_100q(total_cost_usd: float, n_queries: int) -> float:
    if n_queries == 0:
        return 0.0
    return total_cost_usd * 100.0 / n_queries
