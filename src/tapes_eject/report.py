"""Act 2: the questions the labels answer, computed over data/ locally.

Which model or project carries each label, as a rate over every exported session for that model
or project, and how each label moves week over week.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from .curate import labels_by_session, turns_by_session

DIMS = ("model", "project", "week")
CORRECTION_VIEW_LABELS = ("pushback", "observation", "missing-knowledge", "apology")


def week_of(started_at: str | None) -> str | None:
    """Monday of the session's week, as YYYY-MM-DD."""
    if not started_at:
        return None
    day = datetime.fromisoformat(started_at.replace("Z", "+00:00")).date()
    return (day - timedelta(days=day.weekday())).isoformat()


def labels_by(
    dim: str, sessions: list[dict], labels: list[dict], only: str | None = None
) -> list[tuple]:
    """(value, label, sessions_with_label, sessions, rate), most-labeled first within a label.
    The denominator is every exported session with that value, labeled or not."""
    if dim not in ("model", "project"):
        raise ValueError(f"dim must be model or project, not {dim!r}")
    totals: dict[str | None, int] = defaultdict(int)
    hits: dict[tuple[str | None, str], int] = defaultdict(int)
    by_label = labels_by_session(labels)
    for s in sessions:
        value = s.get(dim)
        totals[value] += 1
        for name in by_label.get(s["session_id"], ()):
            if only is None or name == only:
                hits[(value, name)] += 1
    rows = [
        (value, name, n, totals[value], round(n / totals[value], 3))
        for (value, name), n in hits.items()
    ]
    return sorted(rows, key=lambda r: (r[1], -r[4], -r[3], str(r[0])))


def labels_by_week(
    sessions: list[dict], labels: list[dict], only: str | None = None
) -> list[tuple]:
    """(week, label, sessions_with_label), oldest week first."""
    by_label = labels_by_session(labels)
    hits: dict[tuple[str, str], int] = defaultdict(int)
    for s in sessions:
        week = week_of(s.get("started_at"))
        if week is None:
            continue
        for name in by_label.get(s["session_id"], ()):
            if only is None or name == only:
                hits[(week, name)] += 1
    return sorted((w, name, n) for (w, name), n in hits.items())


def corrections(sessions: list[dict], turns: list[dict], labels: list[dict]) -> list[dict]:
    """Each turn an engineer corrected, with their words, its model, and its project."""
    meta = {s["session_id"]: s for s in sessions}
    by_turn = {
        (t["session_id"], t["turn_id"]): t for ts in turns_by_session(turns).values() for t in ts
    }
    out: list[dict] = []
    for row in labels:
        key = (row.get("session_id"), row.get("turn_id"))
        if row["label"] not in CORRECTION_VIEW_LABELS or key not in by_turn:
            continue
        s, t = meta.get(key[0], {}), by_turn[key]
        out.append(
            {
                "label": row["label"],
                "model": s.get("model"),
                "project": s.get("project"),
                "started_at": s.get("started_at"),
                "title": s.get("title"),
                "ordinal": t["ordinal"],
                "correction": t.get("user_prompt"),
                "session_id": key[0],
                "turn_id": key[1],
            }
        )
    return sorted(out, key=lambda r: (r["started_at"] or "", r["session_id"], r["ordinal"]))
