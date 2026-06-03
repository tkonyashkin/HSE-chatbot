from __future__ import annotations

import re

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
HR_RE = re.compile(r"^---+\s*$", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
BOLD_STAR_RE = re.compile(r"\*\*([^*\n]+)\*\*")
BOLD_UNDER_RE = re.compile(r"__([^_\n]+)__")
ITALIC_STAR_RE = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
ITALIC_UNDER_RE = re.compile(r"(?<![_\w])_([^_\n]+)_(?![_\w])")
TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|\-]+\|\s*$")
NEWLINE_COLLAPSE_RE = re.compile(r"\n{3,}")


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def convert_tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    headers: list[str] = []
    in_table = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if TABLE_ROW_RE.match(line):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if TABLE_SEP_RE.match(next_line) and not in_table:
                headers = [c.strip() for c in line.strip().strip("|").split("|")]
                in_table = True
                i += 2
                continue
            if in_table:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) == len(headers):
                    pairs = [
                        f"<b>{escape_html(h)}</b>: {escape_html(c)}"
                        for h, c in zip(headers, cells, strict=False)
                        if c
                    ]
                    out.append(" · ".join(pairs))
                else:
                    out.append(escape_html(line))
                i += 1
                continue
        if in_table:
            in_table = False
            headers = []
        out.append(line)
        i += 1
    return "\n".join(out)


def md_to_telegram_html(text: str) -> str:
    if not text:
        return ""

    placeholders: dict[str, str] = {}

    def stash(content: str, kind: str) -> str:
        key = f"\x00{kind}{len(placeholders)}\x00"
        placeholders[key] = content
        return key

    def repl_code_block(m: re.Match[str]) -> str:
        body = escape_html(m.group(1).rstrip("\n"))
        return stash(f"<pre>{body}</pre>", "CB")

    text = CODE_BLOCK_RE.sub(repl_code_block, text)

    def repl_inline(m: re.Match[str]) -> str:
        return stash(f"<code>{escape_html(m.group(1))}</code>", "IC")

    text = INLINE_CODE_RE.sub(repl_inline, text)

    def repl_link(m: re.Match[str]) -> str:
        label = escape_html(m.group(1))
        url = m.group(2).replace('"', "%22")
        return stash(f'<a href="{url}">{label}</a>', "LN")

    text = LINK_RE.sub(repl_link, text)

    text = convert_tables(text)

    text = HR_RE.sub("", text)
    text = HEADING_RE.sub(lambda m: f"<b>{m.group(2)}</b>", text)
    text = BOLD_STAR_RE.sub(r"<b>\1</b>", text)
    text = BOLD_UNDER_RE.sub(r"<b>\1</b>", text)
    text = ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
    text = ITALIC_UNDER_RE.sub(r"<i>\1</i>", text)

    safe_tags: dict[str, str] = {}
    for tag in ("<b>", "</b>", "<i>", "</i>", "<u>", "</u>"):
        key = f"\x01TAG{len(safe_tags)}\x01"
        safe_tags[key] = tag
        text = text.replace(tag, key)
    text = escape_html(text)
    for key, tag in safe_tags.items():
        text = text.replace(key, tag)

    for key, content in placeholders.items():
        text = text.replace(key, content)

    text = NEWLINE_COLLAPSE_RE.sub("\n\n", text).strip()
    return text
