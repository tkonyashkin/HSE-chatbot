from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from parser.llm import LLMClient
from rag.schemas import Answer, RetrievalHit

T = TypeVar("T", bound=BaseModel)


class LLMAnswer(BaseModel):
    text: str = Field(min_length=1)


@dataclass
class GeneratorConfig:
    system_prompt: str
    user_template: str

    @classmethod
    def from_yaml(cls, path: Path) -> "GeneratorConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            system_prompt=raw["system"],
            user_template=raw["user_template"],
        )


@dataclass
class PydanticAILLMClient:
    model: Model

    def extract(self, prompt: str, text: str, schema: type[T]) -> T:
        agent: Agent[None, T] = Agent(
            model=self.model,
            system_prompt=prompt,
            output_type=schema,
        )
        return agent.run_sync(text).output


def format_facts(facts: dict[str, Any]) -> str:
    if not facts:
        return ""
    lines: list[str] = []
    places = facts.get("places") or {}
    if any(places.values()):
        labels = {
            "budget": "бюджет",
            "paid": "платное",
            "target": "целевая квота",
            "special": "особая квота",
            "separate": "отдельная квота",
        }
        parts = [f"{labels[k]}: {v}" for k, v in places.items() if v is not None and k in labels]
        if parts:
            lines.append(f"Места: {', '.join(parts)}")
    tuition = facts.get("tuition_fee_rub_per_year")
    if tuition:
        lines.append(f"Стоимость обучения: {tuition} руб./год")
    min_scores = facts.get("min_scores_by_subject") or {}
    if min_scores:
        parts = [f"{subj} {score}" for subj, score in min_scores.items()]
        lines.append(f"Минимальные баллы по предметам: {', '.join(parts)}")
    passing_scores = facts.get("passing_scores") or []
    if passing_scores:
        parts = []
        for ps in passing_scores:
            year = ps.get("year")
            budget = ps.get("budget")
            paid = ps.get("paid")
            entry = f"{year}: бюджет {budget}" if budget is not None else f"{year}"
            if paid is not None:
                entry += f", платное {paid}"
            parts.append(entry)
        lines.append(f"Проходные баллы: {'; '.join(parts)}")
    if not lines:
        return ""
    return "Структурированные данные программы:\n" + "\n".join(f"- {line}" for line in lines)


def build_context(hits: list[RetrievalHit]) -> str:
    blocks: list[str] = []
    for hit in hits:
        c = hit.chunk
        header = (
            f"[Источник {hit.rank}]\n"
            f"Программа: {c.program_name} ({c.program_code})\n"
            f"Факультет: {c.faculty}\n"
            f"Раздел: {c.chunk_type.value}\n"
            f"Год приёма: {c.admission_year}\n"
            f"URL: {c.url}\n"
            f"---"
        )
        facts_block = format_facts(c.extracted_facts)
        body_parts = [c.text]
        if facts_block:
            body_parts.append(facts_block)
        blocks.append(f"{header}\n" + "\n\n".join(body_parts))
    return "\n\n".join(blocks)


@dataclass
class Generator:
    llm: LLMClient
    config: GeneratorConfig

    def generate(self, query: str, hits: list[RetrievalHit]) -> Answer:
        context = build_context(hits)
        full_prompt = self.config.system_prompt
        text = self.config.user_template.format(context=context, query=query)
        result = self.llm.extract(prompt=full_prompt, text=text, schema=LLMAnswer)
        return Answer(
            text=result.text,
            source_urls=[str(h.chunk.url) for h in hits],
            retrieved_chunk_ids=[h.chunk.chunk_id for h in hits],
        )
