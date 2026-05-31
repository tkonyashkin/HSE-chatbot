def hit_at_k(retrieved_keys: list[str], relevant_keys: set[str], k: int = 5) -> int:
    return int(any(key in relevant_keys for key in retrieved_keys[:k]))


def reciprocal_rank(retrieved_keys: list[str], relevant_keys: set[str], k: int = 10) -> float:
    for rank, key in enumerate(retrieved_keys[:k], start=1):
        if key in relevant_keys:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_keys: list[str], relevant_keys: set[str], k: int = 20) -> float:
    if not relevant_keys:
        return 0.0
    top_set = set(retrieved_keys[:k])
    return len(top_set & relevant_keys) / len(relevant_keys)
