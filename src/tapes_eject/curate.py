"""Labels decide the data. Pure functions over the rows `export` wrote."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import CORRECTION_LABELS, GOLDEN, NEGATIVE_LABELS, REGRESSION

MAX_CHARS = 6_000  # per message
MAX_TRAIN_TURNS = 12
MAX_CONTEXT_TURNS = 6


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def clip(text: str, n: int = MAX_CHARS) -> str:
    return text if len(text) <= n else text[:n] + "\n[...]"


def labels_by_session(labels: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in labels:
        if row.get("session_id"):
            out.setdefault(row["session_id"], set()).add(row["label"])
    return out


def turns_by_session(turns: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in sorted(turns, key=lambda r: (r["session_id"], r["ordinal"])):
        if not row.get("synthetic"):
            out.setdefault(row["session_id"], []).append(row)
    return out


def conversation(turns: list[dict]) -> list[dict]:
    msgs: list[dict] = []
    for row in turns:
        if row.get("user_prompt"):
            msgs.append({"role": "user", "content": clip(row["user_prompt"])})
        if row.get("agent_text"):
            msgs.append({"role": "assistant", "content": clip(row["agent_text"])})
    return msgs


def is_holdout(session_id: str) -> bool:
    """One `golden` session in five is an eval case instead of training data. Stable across runs,
    so a session never moves between the two and the tuned model is never scored on text it
    was trained on."""
    return int(hashlib.sha1(session_id.encode()).hexdigest()[:8], 16) % 5 == 0


def _trainable(sid: str, labels: set[str], has_outcome: bool | None) -> bool:
    if REGRESSION in labels or labels & set(CORRECTION_LABELS):
        return False  # a known failure or a corrected turn never becomes training data
    if GOLDEN in labels:
        return not is_holdout(sid)
    return has_outcome is True and not labels & NEGATIVE_LABELS


def training_examples(sessions: list[dict], turns: list[dict], labels: list[dict]) -> list[dict]:
    by_label, by_turns = labels_by_session(labels), turns_by_session(turns)
    out: list[dict] = []
    for s in sessions:
        sid = s["session_id"]
        if not _trainable(sid, by_label.get(sid, set()), s.get("has_outcome")):
            continue
        msgs = conversation(by_turns.get(sid, [])[:MAX_TRAIN_TURNS])
        while msgs and msgs[-1]["role"] != "assistant":
            msgs.pop()
        if any(m["role"] == "user" for m in msgs) and msgs:
            out.append({"session_id": sid, "messages": msgs})
    return out


def _correction_turns(labels: list[dict]) -> set[tuple[str, str]]:
    return {
        (r["session_id"], r["turn_id"])
        for r in labels
        if r["label"] in CORRECTION_LABELS and r.get("session_id") and r.get("turn_id")
    }


def correction_cases(turns: list[dict], labels: list[dict]) -> list[dict]:
    """The model sees the conversation up to the turn the engineer corrected, and is judged on
    whether its answer already does what the engineer later had to ask for."""
    by_turns = turns_by_session(turns)
    out: list[dict] = []
    for sid, tid in sorted(_correction_turns(labels)):
        ts = by_turns.get(sid, [])
        pos = next((i for i, row in enumerate(ts) if row["turn_id"] == tid), None)
        if not pos:  # not exported, or the first turn: no earlier agent turn to judge
            continue
        msgs = conversation(ts[max(0, pos - MAX_CONTEXT_TURNS) : pos])
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]  # the model writes that turn itself
        if not msgs:
            continue
        correction = clip(ts[pos]["user_prompt"], 2_000)
        out.append(
            {
                "inputs": {"messages": msgs},
                "expectations": {
                    "guidelines": [
                        "The response already does what the engineer later had to ask for, "
                        f"without being told: {correction}"
                    ]
                },
            }
        )
    return out


def session_cases(sessions: list[dict], turns: list[dict], labels: list[dict]) -> list[dict]:
    by_label, by_turns = labels_by_session(labels), turns_by_session(turns)
    corrections = _correction_turns(labels)
    out: list[dict] = []
    for s in sessions:
        sid = s["session_id"]
        names = by_label.get(sid, set())
        ts = by_turns.get(sid, [])
        if not ts or not ts[0].get("user_prompt") or not names & {GOLDEN, REGRESSION}:
            continue
        request = [{"role": "user", "content": clip(ts[0]["user_prompt"])}]
        if REGRESSION in names:
            said = [clip(r["user_prompt"], 500) for r in ts if (sid, r["turn_id"]) in corrections]
            failure = " | ".join(said[:3]) or s.get("title") or "the earlier failed attempt"
            guideline = (
                "The response must not repeat the failure this request once led to. "
                f"The engineer had to say: {failure}"
            )
        elif ts[0].get("agent_text") and is_holdout(sid):
            guideline = (
                "The response takes the same approach as this known-good answer: "
                f"{clip(ts[0]['agent_text'], 2_000)}"
            )
        else:
            continue
        out.append({"inputs": {"messages": request}, "expectations": {"guidelines": [guideline]}})
    return out


def eval_records(sessions: list[dict], turns: list[dict], labels: list[dict]) -> list[dict]:
    """Every case, one per distinct input. Two cases with the same inputs would score the same
    prompt twice, so merge their guidelines into one case."""
    merged: dict[str, dict] = {}
    for case in correction_cases(turns, labels) + session_cases(sessions, turns, labels):
        key = json.dumps(case["inputs"], sort_keys=True)
        if key not in merged:
            merged[key] = {"inputs": case["inputs"], "expectations": {"guidelines": []}}
        guidelines = merged[key]["expectations"]["guidelines"]
        for g in case["expectations"]["guidelines"]:
            if g not in guidelines:
                guidelines.append(g)
    return list(merged.values())


def counts(sessions: list[dict], turns: list[dict], labels: list[dict]) -> dict[str, int]:
    by_label = labels_by_session(labels)
    corr = len(correction_cases(turns, labels))
    return {
        "sessions": len(sessions),
        "with_outcome": sum(1 for s in sessions if s.get("has_outcome") is True),
        "outcome_unknown": sum(1 for s in sessions if s.get("has_outcome") is None),
        "training_examples": len(training_examples(sessions, turns, labels)),
        "correction_cases": corr,
        "golden": sum(1 for names in by_label.values() if GOLDEN in names),
        "regression": sum(1 for names in by_label.values() if REGRESSION in names),
        "eval_cases": len(eval_records(sessions, turns, labels)),
    }
