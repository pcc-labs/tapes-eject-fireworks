"""Shared fixtures: a scripted paperctl and minimal export records."""

from __future__ import annotations

import json


class FakeRunner:
    """Answers paperctl argv by the first rule whose tokens all appear in it."""

    def __init__(self, rules: list[tuple[tuple[str, ...], object]]):
        self.rules = rules
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], timeout: int) -> str:
        self.calls.append(argv)
        for tokens, answer in self.rules:
            if all(t in argv for t in tokens):
                if isinstance(answer, Exception):
                    raise answer
                return answer if isinstance(answer, str) else json.dumps(answer)
        raise AssertionError(f"unexpected paperctl call: {argv}")


def record(session_id: str, turns: list[tuple[str, str, str]], title: str = "t") -> dict:
    """An export record. turns: (trace_id, user_prompt, assistant_text). Each turn has one llm span
    whose id is `spn_<trace_id>`."""
    return {
        "schema": "2026-06-15",
        "session": {
            "id": session_id,
            "harness_id": "claude",
            "display_title": title,
            "started_at": "2026-09-01T00:00:00Z",
            "rollup": {
                "status": "completed",
                "turn_count": len(turns),
                "model": "claude-opus-5-5",
                "usage": {"cost_usd": 1.5},
            },
        },
        "traces": [
            {
                "trace": {
                    "trace_id": tid,
                    "user_prompt": prompt,
                    "response_preview": reply[:40],
                    "status": "ok",
                    "started_at": f"2026-09-01T00:{i:02d}:00Z",
                },
                "spans": [
                    {
                        "span_id": f"spn_{tid}",
                        "kind": "llm",
                        "seq": 0,
                        "output": [{"type": "text", "text": reply}],
                    }
                ],
            }
            for i, (tid, prompt, reply) in enumerate(turns)
        ],
    }
