from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from rag.agent.llm_adapter import build_pydantic_ai_model


class Intent(StrEnum):
    single_program_details = "single_program_details"
    program_search_broad = "program_search_broad"
    compare_programs = "compare_programs"
    match_by_subject_scores = "match_by_subject_scores"
    analytical_query = "analytical_query"
    admission_deadlines = "admission_deadlines"
    knowledge_lookup = "knowledge_lookup"
    off_topic_refuse = "off_topic_refuse"


class PlanStep(BaseModel):
    intent: Intent
    focus: str


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=3)


def load_planner_prompt(path: Path) -> str:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(raw["system"])


def build_planner(prompt_path: Path) -> Agent[None, Plan]:
    system_prompt = load_planner_prompt(prompt_path)
    model_name = os.environ.get("PLANNER_MODEL")
    model = build_pydantic_ai_model(model_override=model_name)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=Plan,
    )


async def build_plan(query: str, prompt_path: Path) -> Plan:
    agent = build_planner(prompt_path)
    result = await agent.run(query)
    return result.output


def format_plan_for_worker(plan: Plan) -> str:
    lines = ["[План от планировщика]"]
    lines.append("Шаги:")
    for i, step in enumerate(plan.steps, start=1):
        line = f"  {i}. intent={step.intent.value}, focus={step.focus}"
        lines.append(line)
    return "\n".join(lines)
