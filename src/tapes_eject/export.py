"""Paper -> redacted rows. Labels, including `no-outcome`, come from Paper; nothing here calls
the autolabel cassette.

Session text is parsed and redacted in session.py, then scrubbed again here for JSON-quoted keys,
URL credentials, and a few vendor token shapes. Redaction is pattern-based: it lowers the risk of
a secret leaving this machine, it does not remove it. Raw records are only kept in the local
export cache (data/cache/, git-ignored) and are never uploaded.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import DEFAULT_MAX_OUTPUT_TOKENS, NO_OUTCOME, Config
from .paper import PaperError
from .session import Session, parse_session

LEVELS = ("session", "trace", "span")
RECENT_MIN_TURNS = 1  # the size caps bound the other end


@dataclass
class Export:
    sessions: list[dict]
    turns: list[dict]
    labels: list[dict]
    failed: list[tuple[str, str]] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    outcome_unknown: int = 0

    @property
    def outcome_known(self) -> bool:
        return self.outcome_unknown == 0


REDACTED = "[redacted]"
_EXTRA_SECRETS = [
    # JSON-quoted secrets: {"api_key": "..."}
    (
        re.compile(
            r'(?i)("[\w-]*(?:api[_-]?key|secret|token|password|passwd)"\s*:\s*")[^"]{8,}(")'
        ),
        rf"\1{REDACTED}\2",
    ),
    # URL userinfo: scheme://user:password@host
    (re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]+(@)"), rf"\1{REDACTED}\2"),
    # dapi-prefixed personal access tokens
    (re.compile(r"\bdapi[0-9a-f]{32}\b"), REDACTED),
    # Stripe-style keys
    (re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}"), REDACTED),
]


def scrub(text: str) -> str:
    """A second pass for secret shapes session.redact does not cover."""
    for pattern, repl in _EXTRA_SECRETS:
        text = pattern.sub(repl, text)
    return text


def index_record(record: dict) -> tuple[str, dict[str, str], dict[str, str]]:
    sid = (record.get("session") or {}).get("id") or ""
    trace_session: dict[str, str] = {}
    span_trace: dict[str, str] = {}
    for entry in record.get("traces") or []:
        tid = (entry.get("trace") or {}).get("trace_id")
        if not tid:
            continue
        trace_session[tid] = sid
        for span in entry.get("spans") or []:
            if span.get("span_id"):
                span_trace[span["span_id"]] = tid
    return sid, trace_session, span_trace


def label_rows(
    attachments: dict[str, list[dict]],
    trace_session: dict[str, str],
    span_trace: dict[str, str],
    evidence: dict[tuple[str, str], str] | None = None,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    unmapped: list[dict] = []
    for label, atts in attachments.items():
        for a in atts:
            ptype, pid = a["primitive_type"], a["primitive_id"]
            sid = tid = spid = None
            if ptype == "session":
                sid = pid
            elif ptype == "trace":
                tid = pid
                sid = trace_session.get(pid)
            elif ptype == "span":
                spid = pid
                # Paper names a span `<trace id>~<span id>`; older ids are bare span ids.
                tid = pid.split("~", 1)[0] if "~" in pid else span_trace.get(pid)
                sid = trace_session.get(tid) if tid else None
            else:
                continue  # skills and other primitives are not session data
            row = {
                "label": label,
                "primitive_type": ptype,
                "primitive_id": pid,
                "session_id": sid,
                "turn_id": tid,
                "span_id": spid,
                "evidence": (evidence or {}).get((label, tid)) if tid else None,
            }
            rows.append(row)
            if sid is None:
                unmapped.append(row)
    return rows, unmapped


def project_name(cwd: str | None) -> str | None:
    """The last path segment of the session's working directory: the repo, without the home
    directory or username above it."""
    name = (cwd or "").rstrip("/").rsplit("/", 1)[-1]
    return name or None


def session_row(sess: Session, no_outcome: set[str], unknown: set[str] = frozenset()) -> dict:
    return {
        "session_id": sess.id,
        "title": scrub(sess.title),
        "project": project_name(sess.cwd),
        "harness": sess.harness,
        "model": sess.model,
        "started_at": sess.started_at,
        "status": sess.status,
        "turn_count": sess.turn_count,
        "cost_usd": sess.cost_usd,
        "has_outcome": False if sess.id in no_outcome else (None if sess.id in unknown else True),
    }


def turn_rows(sess: Session) -> list[dict]:
    return [
        {
            "session_id": sess.id,
            "turn_id": t.id,
            "ordinal": t.index,
            "user_prompt": scrub(t.user_prompt),
            "agent_text": scrub("\n\n".join(t.assistant_text) or t.response_preview),
            "synthetic": t.synthetic,
        }
        for t in sess.turns
    ]


def _turns(item: dict) -> int:
    return int((item.get("rollup") or {}).get("turn_count") or 0)


def _output_tokens(item: dict) -> int:
    usage = (item.get("rollup") or {}).get("usage") or {}
    return int(usage.get("output_tokens") or 0)


def too_big(
    item: dict,
    max_turns: int,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    skip: frozenset[str] = frozenset(),
) -> str | None:
    """Why this session list item must not be exported, or None. Judged from the list rollup
    alone, so no export request is made for a session that would be dropped anyway. Turn count
    misses a short session with enormous tool output; the rollup's output tokens catch it."""
    if item.get("id") in skip:
        return "in TAPES_EJECT_SKIP_SESSIONS: its export overloads Paper"
    if _turns(item) > max_turns:
        return f"{_turns(item)} turns > max {max_turns}"
    if _output_tokens(item) > max_output_tokens:
        return f"{_output_tokens(item)} output tokens > max {max_output_tokens}"
    return None


def choose_sessions(
    labeled: dict[str, dict],
    recent: list[dict],
    sample: int,
    max_turns: int,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    skip: frozenset[str] = frozenset(),
) -> tuple[list[str], list[tuple[str, str]]]:
    """Every labeled session that is not too big, plus recent non-empty ones under the caps, so
    per-model and per-project rates have every session the labeler saw as their denominator."""
    ids: list[str] = []
    skipped: list[tuple[str, str]] = []
    for sid, it in labeled.items():
        reason = too_big(it, max_turns, max_output_tokens, skip)
        if reason:
            skipped.append((sid, reason))
        else:
            ids.append(sid)
    extra = [
        it["id"]
        for it in recent
        if it["id"] not in labeled
        and _turns(it) >= RECENT_MIN_TURNS
        and too_big(it, max_turns, max_output_tokens, skip) is None
    ]
    return ids + extra[:sample], skipped


def _from_cache(sid: str, seen: str | None, cache_dir: Path | None) -> dict | None:
    """The cached record when Paper says the session has not changed since, else None."""
    path = cache_dir / f"{sid}.json" if cache_dir and seen else None
    if path and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("last_seen_at") == seen:
            return cached["record"]
    return None


def _cached_export(
    paper, sid: str, seen: str | None, cache_dir: Path | None
) -> tuple[dict | None, bool]:
    """(the session's record, whether Paper was asked for it)."""
    cached = _from_cache(sid, seen, cache_dir)
    if cached is not None:
        return cached, False
    path = cache_dir / f"{sid}.json" if cache_dir and seen else None
    rec = paper.export_session(sid)
    if rec and path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_seen_at": seen, "record": rec}), encoding="utf-8")
    return rec, True


OUTAGE_SIGNS = ("could not reach", "timed out", "truncated")


def is_outage(err: Exception) -> bool:
    """An export failure that means the service is down, not that this session is bad. Carrying
    on would pile requests onto a service that is already struggling."""
    text = str(err)
    return any(sign in text for sign in OUTAGE_SIGNS)


def problems(report: dict) -> list[str]:
    """Why an export should not be synced as-is. Empty means it is safe to sync."""
    out: list[str] = []
    if report.get("failed"):
        out.append(f"{len(report['failed'])} session exports failed")
    if not report.get("sessions"):
        out.append("no sessions exported")
    if report.get("outcome_unknown"):
        out.append(f"outcome unknown for {report['outcome_unknown']} sessions")
    return out


def run_export(
    paper,
    cfg: Config,
    log: Callable[[str], None] = print,
    cache_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Export:
    labels = [lab for lab in paper.labels() if any((lab.get("usage") or {}).get(t) for t in LEVELS)]
    attachments = {
        lab["name"]: [
            a
            for t in LEVELS
            if (lab.get("usage") or {}).get(t)
            for a in paper.attachments(lab["id"], t)
        ]
        for lab in labels
    }
    labeled: dict[str, dict] = {}
    for lab in labels:
        if (lab.get("usage") or {}).get("session"):
            for it in paper.sessions(label=lab["name"], limit=10_000):
                labeled[it["id"]] = it
    since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent = paper.sessions(since=since, limit=cfg.sample_sessions * 3)
    ids, skipped = choose_sessions(
        labeled,
        recent,
        cfg.sample_sessions,
        cfg.max_turns,
        cfg.max_output_tokens,
        cfg.skip_sessions,
    )
    items = {**{it["id"]: it for it in recent}, **labeled}
    failed: list[tuple[str, str]] = []

    parsed: dict[str, Session] = {}
    trace_session: dict[str, str] = {}
    span_trace: dict[str, str] = {}
    outage = False  # after an outage-shaped failure, only the cache is read; no more requests
    for i, sid in enumerate(ids, 1):
        seen = items.get(sid, {}).get("last_seen_at")
        if outage:
            rec = _from_cache(sid, seen, cache_dir)
            if rec is None:
                failed.append((sid, "not tried: export service unreachable"))
                continue
            fetched = False
        else:
            log(f"[{i}/{len(ids)}] export {sid}")
            try:
                rec, fetched = _cached_export(paper, sid, seen, cache_dir)
            except PaperError as e:
                failed.append((sid, str(e)))
                if is_outage(e):
                    outage = True
                    log("export service unreachable; reading the rest from the cache only")
                continue
        if fetched and cfg.export_pause > 0:
            sleep(cfg.export_pause)
        if not rec:
            continue
        _, ts, st = index_record(rec)
        trace_session.update(ts)
        span_trace.update(st)
        sess = parse_session(rec)
        if sess is not None:
            parsed[sess.id] = sess

    # Paper's `no-outcome` label is the record of a session that produced nothing. A session
    # without it is taken to have an outcome: labels are an input here, not something to
    # recompute.
    no_outcome = {
        a["primitive_id"]
        for a in attachments.get(NO_OUTCOME, [])
        if a["primitive_type"] == "session"
    }

    rows, unmapped = label_rows(attachments, trace_session, span_trace)
    return Export(
        sessions=[session_row(s, no_outcome) for s in parsed.values()],
        turns=[r for s in parsed.values() for r in turn_rows(s)],
        labels=rows,
        failed=failed,
        unmapped=unmapped,
        skipped=skipped,
        outcome_unknown=0,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_export(ex: Export, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_dir / "sessions.jsonl", ex.sessions)
    _write_jsonl(data_dir / "turns.jsonl", ex.turns)
    _write_jsonl(data_dir / "labels.jsonl", ex.labels)
    report = {
        "sessions": len(ex.sessions),
        "turns": len(ex.turns),
        "labels": len(ex.labels),
        "failed": ex.failed,
        "skipped": ex.skipped,
        "unmapped": len(ex.unmapped),
        "unmapped_by_label": _count(r["label"] for r in ex.unmapped),
        "outcome_unknown": ex.outcome_unknown,
    }
    (data_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
