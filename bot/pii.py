from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class PIIType(StrEnum):
    email = "EMAIL"
    phone = "PHONE"
    passport = "PASSPORT"
    snils = "SNILS"
    inn = "INN"
    card = "CARD"
    birthdate = "BIRTHDATE"
    diploma = "DIPLOMA"
    person = "PERSON"


@dataclass(frozen=True)
class PIISpan:
    start: int
    end: int
    pii_type: PIIType
    text: str
    confidence: float
    source: str


@dataclass
class DetectionResult:
    masked_text: str
    spans: list[PIISpan] = field(default_factory=list)

    def has_pii(self) -> bool:
        return bool(self.spans)


EMAIL_RE = re.compile(r"[\w.%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", re.UNICODE)


PHONE_RE = re.compile(
    r"(?:"
    r"\+7|8(?!0)"
    r")[\s\-]*\(?\s*\d{3}\s*\)?[\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}"
    r"|"
    r"\+375[\s\-]*\(?\s*\d{2}\s*\)?[\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}"
    r"|"
    r"80\d{9}"
)

PASSPORT_RE = re.compile(r"\b\d{4}[\s\-]?\d{6}\b")
SNILS_RE = re.compile(r"\b\d{3}[\-\s]\d{3}[\-\s]\d{3}\s\d{2}\b")
INN_RE = re.compile(r"(?<!\d)\d{10}(?!\d)|(?<!\d)\d{12}(?!\d)")
CARD_RE = re.compile(r"\b(?:\d{4}[\s\-]?){3,4}\d{1,4}\b")
BIRTHDATE_RE = re.compile(
    r"\b(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./](?:19[4-9]\d|20[01]\d)\b"
)
DIPLOMA_RE = re.compile(r"\b[А-ЯA-Z]{2}\s?\d{14}\b")


def luhn_valid(card_digits: str) -> bool:
    digits = [int(d) for d in card_digits if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def inn_valid(inn_digits: str) -> bool:
    digits = [int(d) for d in inn_digits if d.isdigit()]
    if len(digits) == 10:
        coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        s = sum(d * c for d, c in zip(digits[:9], coeffs, strict=True))
        return s % 11 % 10 == digits[9]
    if len(digits) == 12:
        coeffs_11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coeffs_12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        c11 = sum(d * c for d, c in zip(digits[:10], coeffs_11, strict=True)) % 11 % 10
        c12 = sum(d * c for d, c in zip(digits[:11], coeffs_12, strict=True)) % 11 % 10
        return c11 == digits[10] and c12 == digits[11]
    return False


def regex_detect(text: str) -> list[PIISpan]:
    spans: list[PIISpan] = []

    for m in EMAIL_RE.finditer(text):
        spans.append(PIISpan(m.start(), m.end(), PIIType.email, m.group(0), 0.99, "regex"))

    for m in PHONE_RE.finditer(text):
        spans.append(PIISpan(m.start(), m.end(), PIIType.phone, m.group(0), 0.95, "regex"))

    for m in PASSPORT_RE.finditer(text):
        ctx_start = max(0, m.start() - 30)
        ctx_end = min(len(text), m.end() + 30)
        if "паспорт" not in text[ctx_start:ctx_end].lower():
            continue
        spans.append(
            PIISpan(m.start(), m.end(), PIIType.passport, m.group(0), 0.95, "regex+context")
        )

    for m in SNILS_RE.finditer(text):
        spans.append(PIISpan(m.start(), m.end(), PIIType.snils, m.group(0), 0.95, "regex"))

    for m in INN_RE.finditer(text):
        if inn_valid(m.group(0)):
            spans.append(
                PIISpan(m.start(), m.end(), PIIType.inn, m.group(0), 0.95, "regex+checksum")
            )

    for m in CARD_RE.finditer(text):
        digits_only = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits_only) <= 19 and luhn_valid(digits_only):
            spans.append(PIISpan(m.start(), m.end(), PIIType.card, m.group(0), 0.95, "regex+luhn"))

    for m in BIRTHDATE_RE.finditer(text):
        ctx_start = max(0, m.start() - 40)
        ctx = text[ctx_start : m.start()].lower()
        if any(k in ctx for k in ("рожден", "родил", "дата рожд", "д.р.", "др.")):
            spans.append(
                PIISpan(m.start(), m.end(), PIIType.birthdate, m.group(0), 0.90, "regex+context")
            )

    for m in DIPLOMA_RE.finditer(text):
        spans.append(PIISpan(m.start(), m.end(), PIIType.diploma, m.group(0), 0.85, "regex"))

    return spans


class NERPipeline(Protocol):
    def __call__(self, text: str) -> list[dict[str, Any]]: ...


def ner_detect(text: str, ner_pipeline: NERPipeline | None) -> list[PIISpan]:
    if ner_pipeline is None:
        return []
    try:
        entities = ner_pipeline(text)
    except Exception:
        return []

    type_map = {
        "PER": PIIType.person,
    }
    spans: list[PIISpan] = []
    for ent in entities:
        raw_label = str(ent.get("entity_group") or ent.get("entity") or "").upper()
        label = raw_label.replace("B-", "").replace("I-", "")
        pii_type = type_map.get(label)
        if pii_type is None:
            continue
        score = float(ent.get("score", 0.0))
        if score < 0.4:
            continue
        start = int(ent["start"])
        end = int(ent["end"])
        spans.append(PIISpan(start, end, pii_type, text[start:end], score, "ner"))
    return spans


def resolve_overlaps(spans: list[PIISpan]) -> list[PIISpan]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: (-s.confidence, s.start - s.end))
    kept: list[PIISpan] = []
    for s in sorted_spans:
        overlaps = any(not (s.end <= k.start or s.start >= k.end) for k in kept)
        if not overlaps:
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def apply_mask(text: str, spans: list[PIISpan]) -> str:
    result = text
    for s in sorted(spans, key=lambda x: -x.start):
        result = result[: s.start] + f"[{s.pii_type.value}]" + result[s.end :]
    return result


class PIIDetector:
    def __init__(
        self,
        ner_pipeline: NERPipeline | None = None,
        enable_ner: bool = True,
    ):
        self.ner_pipeline = ner_pipeline if enable_ner else None

    def detect(self, text: str) -> DetectionResult:
        if not text or len(text.strip()) == 0:
            return DetectionResult(masked_text=text, spans=[])
        regex_spans = regex_detect(text)
        ner_spans = ner_detect(text, self.ner_pipeline)
        merged = resolve_overlaps(regex_spans + ner_spans)
        masked = apply_mask(text, merged)
        return DetectionResult(masked_text=masked, spans=merged)

    async def detect_async(self, text: str) -> DetectionResult:
        return self.detect(text)

    def mask(self, text: str) -> str:
        return self.detect(text).masked_text


def build_default_detector(enable_ner: bool = True) -> PIIDetector:
    ner_pipeline: NERPipeline | None = None
    if enable_ner:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_name = "Davlan/xlm-roberta-base-ner-hrl"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        ner_pipeline = pipeline(  # type: ignore[call-overload]
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    return PIIDetector(ner_pipeline=ner_pipeline, enable_ner=enable_ner)
