from __future__ import annotations

import re


def expand_acronyms_in_query(text: str, acronyms: dict[str, str]) -> str:
    out = text
    for acr in sorted(acronyms.keys(), key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(acr) + r"\b", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(f"{acr} ({acronyms[acr]})", out)
    return out
