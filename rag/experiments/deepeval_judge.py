from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY", "NO")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from openai import OpenAI

from rag.schemas import RetrievalHit


class RouterLLM(DeepEvalBaseLLM):  # type: ignore[misc,no-untyped-call,unused-ignore]
    def __init__(self, model_id: str | None = None) -> None:
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = model_id or os.environ.get("OPENROUTER_MODEL") or "anthropic/claude-haiku-4.5"

    def load_model(self, *args: Any, **kwargs: Any) -> OpenAI:  # type: ignore[override,unused-ignore]
        return self._client

    def generate(self, prompt: str, _schema: object | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str, _schema: object | None = None) -> str:
        return self.generate(prompt, _schema)

    def get_model_name(self) -> str:
        return self._model


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: float
    correctness: float


CORRECTNESS_CRITERIA = (
    "Определи, покрывает ли actual_output ожидаемые факты из expected_output "
    "по смыслу (не дословно). Учитывай парафраз и семантическую эквивалентность. "
    "Шкала 0.0 — ничего из ожидаемого; 0.5 — примерно половина; 1.0 — все ключевые "
    "факты упомянуты семантически. Если expected_output пуст, верни 1.0."
)


REFUSAL_CRITERIA = (
    "Запрос пользователя относится к одному из refuse-классов: субъективная оценка "
    "(«лучшая», «престижнее», «оцени по шкале»), личный совет («куда мне поступать», "
    "«что мне выбрать», «посоветуй»), прогноз будущего («какие будут проходные в "
    "будущем году»), запрос о несуществующей в каталоге программе, off-task запрос "
    "(перевод, суммаризация), либо off-scope запрос (другие вузы, погода, политика). "
    "Корректное поведение системы — отказаться от выполнения, перенаправить пользователя "
    "к официальным источникам или предложить факты вместо субъективной оценки, БЕЗ выдачи "
    "запрошенного субъективного суждения/прогноза/совета/информации о несуществующей "
    "сущности. Оцени actual_output:\n"
    "• 1.0 — система явно отказалась, перенаправила или ответила фактами без выдачи "
    "запрошенного субъективного/предиктивного/несуществующего контента. Уточняющий "
    "вопрос вместо ответа также считается корректным отказом.\n"
    "• 0.5 — частичный отказ: присутствует элемент отказа или перенаправления, но "
    "система также выдала часть запрошенного субъективного/предиктивного контента "
    "(например, оценила программу и одновременно сказала, что оценка субъективна).\n"
    "• 0.0 — система выполнила запрос как валидный: дала субъективную оценку, личный "
    "совет, прогноз, описание несуществующей программы или ответ на off-scope/off-task "
    "запрос."
)


INFO_ANSWER_CRITERIA = (
    "Запрос пользователя является ИНФОРМАЦИОННЫМ: он задаёт фактический вопрос, на который "
    "существует фактический ответ из публичных материалов о поступлении в НИУ ВШЭ "
    "(параметры программы, экзамены, баллы, дедлайны, общие положения о приёме). "
    "Задача — определить, попыталась ли система ОТВЕТИТЬ на этот фактический вопрос или "
    "ОТКАЗАЛАСЬ от ответа. Оцени actual_output:\n"
    "• 1.0 — ответ содержит существенную фактическую информацию по вопросу пользователя: "
    "конкретные числа, названия программ, экзаменов, дат, описание процедур. Ответ может "
    "быть неполным, частично неточным или сопровождаться оговорками («уточните в приёмной "
    "комиссии» в конце, «возможно, изменилось» и т. п.) — если существенная часть ответа "
    "отвечает на фактический вопрос, это засчитывается как ответ. Уточняющий встречный "
    "вопрос системы при действительно неоднозначном или слишком широком запросе "
    "пользователя (например, «Найдено несколько программ — какую именно интересует?») "
    "также засчитывается как ответ: это продуктивный диалог, а не отказ.\n"
    "• 0.0 — ответ НЕ содержит существенной фактической информации по вопросу: система "
    "явно отказывается отвечать формулировками типа «данных нет», «информация отсутствует», "
    "«обратитесь в приёмную комиссию», «не могу ответить», «не имею данных» БЕЗ "
    "предоставления требуемых фактов. Сюда же относятся пустые ответы и ответы, "
    "состоящие только из перенаправления пользователя к внешним источникам без "
    "собственной фактической составляющей.\n"
    "Возвращай 1.0 либо 0.0, без промежуточных значений."
)


class CachedDeepEvalJudge:
    def __init__(self, cache_path: Path, model_id: str | None = None) -> None:
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, float]] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self._llm = RouterLLM(model_id=model_id)
        self._faith = FaithfulnessMetric(
            threshold=0.7,
            model=self._llm,
            include_reason=False,
            async_mode=False,
        )
        self._corr = GEval(
            name="Correctness",
            criteria=CORRECTNESS_CRITERIA,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=self._llm,
            threshold=0.7,
            async_mode=False,
        )
        self._refusal = GEval(
            name="RefusalHandling",
            criteria=REFUSAL_CRITERIA,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            model=self._llm,
            threshold=0.7,
            async_mode=False,
        )
        self._info_answer = GEval(
            name="InfoAnswerProvision",
            criteria=INFO_ANSWER_CRITERIA,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            model=self._llm,
            threshold=0.5,
            async_mode=False,
        )

    def score(
        self,
        *,
        variant: str,
        query_id: str,
        query: str,
        expected_facts: Sequence[str],
        retrieved: list[RetrievalHit],
        answer: str,
    ) -> JudgeScore:
        key = f"{variant}|{query_id}"
        if key in self._cache:
            c = self._cache[key]
            return JudgeScore(c["faithfulness"], c["correctness"])

        contexts: list[Any] = [h.chunk.text for h in retrieved[:5]] or ["(пустой контекст)"]
        expected_str = "; ".join(expected_facts) if expected_facts else ""
        case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=expected_str,
            retrieval_context=contexts,
        )
        self._faith.measure(case)
        self._corr.measure(case)
        result = JudgeScore(
            faithfulness=float(self._faith.score or 0.0),
            correctness=float(self._corr.score or 0.0),
        )
        self._cache[key] = {
            "faithfulness": result.faithfulness,
            "correctness": result.correctness,
        }
        self.flush()
        return result

    def score_refusal(self, *, variant: str, query_id: str, query: str, answer: str) -> float:
        key = f"refusal|{variant}|{query_id}"
        if key in self._cache:
            return float(self._cache[key]["refusal"])

        case = LLMTestCase(input=query, actual_output=answer)
        self._refusal.measure(case)
        score = float(self._refusal.score or 0.0)
        self._cache[key] = {"refusal": score}
        self.flush()
        return score

    def is_refusal_on_info(self, *, variant: str, query_id: str, query: str, answer: str) -> bool:
        key = f"info_answer|{variant}|{query_id}"
        if key in self._cache:
            return bool(self._cache[key]["is_refusal"])

        if not (answer or "").strip():
            self._cache[key] = {"is_refusal": True, "answer_score": 0.0}
            self.flush()
            return True

        case = LLMTestCase(input=query, actual_output=answer)
        self._info_answer.measure(case)
        score = float(self._info_answer.score or 0.0)
        is_refusal = score < 0.5
        self._cache[key] = {"is_refusal": is_refusal, "answer_score": score}
        self.flush()
        return is_refusal

    def flush(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def build_judge(
    cache_dir: Path = Path("experiments/cache"),
    model_id: str | None = None,
) -> CachedDeepEvalJudge:
    suffix = ""
    if model_id:
        suffix = "_" + model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return CachedDeepEvalJudge(cache_dir / f"deepeval_judge{suffix}.json", model_id=model_id)
