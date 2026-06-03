from __future__ import annotations


def safe_model_slug(model_id: str) -> str:
    return (
        model_id.replace("/", "_").replace("-", "_").replace(".", "_").replace("#", "_")
    )
