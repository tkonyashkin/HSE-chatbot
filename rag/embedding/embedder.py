from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


@dataclass
class EmbedderConfig:
    model_name: str
    dim: int
    query_prefix: str
    passage_prefix: str
    batch_size: int = 16
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: Path) -> "EmbedderConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = raw["embedding"]
        return cls(
            model_name=cfg["model_name"],
            dim=int(cfg["dim"]),
            query_prefix=cfg["query_prefix"],
            passage_prefix=cfg["passage_prefix"],
            batch_size=int(cfg.get("batch_size", 16)),
            device=cfg.get("device", "cpu"),
        )


class Embedder:
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def encode_passage(self, text: str) -> np.ndarray:
        out = self.model.encode(
            self.config.passage_prefix + text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return out.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        out = self.model.encode(
            self.config.query_prefix + text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return out.astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.config.passage_prefix + t for t in texts]
        out = self.model.encode(
            prefixed,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return out.astype(np.float32)
