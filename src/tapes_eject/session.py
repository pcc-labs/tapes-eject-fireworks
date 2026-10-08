"""One `paperctl sessions export` record -> a Session with its turns, redacted.

Only what `export` writes is parsed: the session's metadata, and per turn the user's prompt, the
agent's text, and whether the turn was synthetic (a resume notice, an interrupt, a keep-alive).
Redaction is pattern-based. It lowers the chance of a secret leaving this machine; it does not
guarantee it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TEXT_CAP = 8_000
REDACTED = "[redacted]"

_SECRETS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), REDACTED),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), REDACTED),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), REDACTED),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
        ),
        REDACTED,
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}"), f"Bearer {REDACTED}"),
    (
        re.compile(
            # Prefixed names too: OPENAI_API_KEY=, GITHUB_TOKEN=, DB_PASSWORD=.
            r"(?i)\b((?:[A-Za-z0-9_\-]*(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token|"
            r"auth[_\-]?token|token|secret|password|passwd))\s*[=:]\s*['\"]?)"
            r"([A-Za-z0-9_\-./+]{12,})"
        ),
        rf"\1{REDACTED}",
    ),
]

# Harness scaffolding injected into the user's prompt; not something the engineer typed.
_SCAFFOLDING = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(
        r"<(command-name|command-message|command-args|local-command-stdout)>.*?</\1>", re.DOTALL
    ),
]
_SYNTHETIC_PREFIXES = (
    "the user stepped away",
    "reply with the single word ok",
    "this session is being continued",
    "[request interrupted",
)


def redact(text: str | None) -> str:
    """Replace secret-shaped substrings. Idempotent."""
    text = text or ""
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def clean_prompt(text: str) -> str:
    for pattern in _SCAFFOLDING:
        text = pattern.sub("", text)
    return text.strip()


def is_synthetic(prompt: str) -> bool:
    p = prompt.strip().lower()
    return not p or p.startswith(_SYNTHETIC_PREFIXES)


@dataclass
class Turn:
    id: str
    index: int
    user_prompt: str
    response_preview: str
    synthetic: bool
    assistant_text: list[str] = field(default_factory=list)


@dataclass
class Session:
    id: str
    title: str
    cwd: str
    harness: str
    model: str
    started_at: str
    status: str
    turn_count: int
    cost_usd: float
    turns: list[Turn] = field(default_factory=list)


def parse_turn(entry: dict, index: int) -> Turn:
    trace = entry.get("trace") or {}
    prompt = clean_prompt(redact(trace.get("user_prompt")))
    turn = Turn(
        id=trace.get("trace_id") or f"turn-{index}",
        index=index,
        user_prompt=prompt,
        response_preview=redact(trace.get("response_preview")),
        synthetic=is_synthetic(prompt),
    )
    for span in sorted(entry.get("spans") or [], key=lambda s: int(s.get("seq") or 0)):
        if span.get("kind") != "llm":
            continue
        for block in span.get("output") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                if isinstance(block.get("text"), str):
                    turn.assistant_text.append(redact(block["text"][:TEXT_CAP]))
    return turn


def parse_session(record: dict) -> Session | None:
    """None for a session with no turns."""
    s = record.get("session") or {}
    rollup = s.get("rollup") or {}
    traces = record.get("traces") or []
    if not traces:
        return None
    sess = Session(
        id=s.get("id") or "",
        title=redact(s.get("display_title") or s.get("name") or rollup.get("preview")),
        cwd=s.get("cwd") or "",
        harness=s.get("harness_id") or "",
        model=rollup.get("model") or "",
        started_at=s.get("started_at") or "",
        status=rollup.get("status") or "unknown",
        turn_count=int(rollup.get("turn_count") or len(traces)),
        cost_usd=float((rollup.get("usage") or {}).get("cost_usd") or 0.0),
    )
    entries = sorted(traces, key=lambda e: (e.get("trace") or {}).get("started_at") or "")
    sess.turns = [parse_turn(entry, i) for i, entry in enumerate(entries)]
    return sess
