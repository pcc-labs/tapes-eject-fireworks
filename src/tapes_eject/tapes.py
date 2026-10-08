"""Sessions from a local tapes stack, and labels kept in a local file.

tapes imports Codex and Claude Code history into Postgres on your machine and serves it at
http://127.0.0.1:18081 (see https://github.com/pcc-labs/tapes-test). Its session records have the
same shape Paper's do, so `export` reads either one. `Tapes` has the same four methods `export`
calls on `Paper`: labels, attachments, sessions, and export_session.

Local tapes has no labels of its own. They live in data/local_labels.jsonl, one row per label on
a session or a turn, written by `tapes-eject label` (found automatically) and `tapes-eject mark`
(set by hand).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable

from .paper import PaperError
from .session import Session, redact

Fetcher = Callable[[str, int], bytes]

AUTO_LABELS = ("pushback", "apology")
MANUAL_LABELS = ("golden", "regression", "no-outcome")

# An engineer correcting the agent. Matched against the start of a later turn's prompt, where a
# correction usually sits ("no, ...", "don't ...", "you deleted ..."), and a few phrases anywhere.
_PUSHBACK_START = re.compile(
    r"^(no\b|nope\b|wrong\b|stop\b|wait\b|don'?t\b|do not\b|that'?s not\b|that is not\b|"
    r"not what\b|actually\b|instead\b|why did you\b|why are you\b|undo\b|revert\b|"
    r"you (changed|deleted|removed|broke|forgot|missed|didn'?t|did not)\b|i said\b|i asked\b|"
    r"please don'?t\b)",
    re.IGNORECASE,
)
_PUSHBACK_ANYWHERE = re.compile(
    r"(that'?s (wrong|not right|incorrect)|not what i (asked|wanted|meant)|"
    r"you (broke|ignored)\b|i told you|as i said)",
    re.IGNORECASE,
)
# The agent backtracking.
_APOLOGY = re.compile(
    r"\b(you'?re (absolutely )?right|you are right|i apologi[sz]e|my apologies|sorry|"
    r"my mistake|i was wrong|good catch)\b",
    re.IGNORECASE,
)
EVIDENCE_CAP = 300


def http_get(url: str, timeout: int) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return b""
        raise PaperError(f"GET {url} failed: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise PaperError(f"GET {url} could not reach tapes: {e}") from e


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_labels(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["label"], r["session_id"], r["primitive_id"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class Tapes:
    def __init__(self, api: str, labels_path: Path, fetch: Fetcher = http_get):
        self.api = api.rstrip("/")
        self.labels_path = labels_path
        self._fetch = fetch

    def _get(self, path: str, timeout: int = 60) -> dict:
        body = self._fetch(f"{self.api}{path}", timeout)
        return json.loads(body) if body else {}

    def _rows(self) -> list[dict]:
        return read_labels(self.labels_path)

    def labels(self) -> list[dict]:
        usage: dict[str, Counter] = {}
        for r in self._rows():
            usage.setdefault(r["label"], Counter())[r["primitive_type"]] += 1
        return [{"id": name, "name": name, "usage": dict(c)} for name, c in usage.items()]

    def attachments(self, label_id: str, primitive_type: str) -> list[dict]:
        return [
            {"primitive_type": r["primitive_type"], "primitive_id": r["primitive_id"]}
            for r in self._rows()
            if r["label"] == label_id and r["primitive_type"] == primitive_type
        ]

    def sessions(
        self, label: str | None = None, since: str | None = None, limit: int = 200
    ) -> list[dict]:
        """Session list items, newest first. With `label`, the sessions carrying it on the session
        or on any of its turns."""
        wanted = {r["session_id"] for r in self._rows() if r["label"] == label} if label else None
        out: list[dict] = []
        cursor: str | None = None
        while True:
            query = {"limit": "200", **({"cursor": cursor} if cursor else {})}
            page = self._get(f"/v1/sessions?{urllib.parse.urlencode(query)}")
            items = page.get("items") or []
            for it in items:
                if since and (it.get("last_seen_at") or "") < since:
                    continue
                if wanted is not None and it["id"] not in wanted:
                    continue
                out.append(it)
            cursor = page.get("next_cursor")
            if not cursor or not items or (wanted is None and len(out) >= limit):
                return out[:limit]

    def export_session(self, session_id: str, timeout: int = 300) -> dict | None:
        """The session's full record: {schema, session, traces: [{trace, spans}], links}."""
        body = self._fetch(
            f"{self.api}/v1/sessions/{urllib.parse.quote(session_id)}/traces", timeout
        )
        if not body:
            return None
        try:
            return json.loads(body)
        except ValueError as e:
            raise PaperError(f"sessions traces {session_id} truncated at {len(body)} bytes") from e


def _evidence(text: str) -> str:
    text = " ".join(redact(text).split())
    return text if len(text) <= EVIDENCE_CAP else text[:EVIDENCE_CAP] + "..."


def is_pushback(prompt: str) -> bool:
    p = prompt.strip()
    return bool(_PUSHBACK_START.match(p) or _PUSHBACK_ANYWHERE.search(p))


def find_labels(sess: Session) -> list[dict]:
    """`pushback` on each turn where the engineer corrects the agent, `apology` on each turn where
    the agent backtracks. Pattern-based: fast, free, and wrong some of the time. Read the
    evidence, and drop or add labels with `mark` where it matters."""
    rows: list[dict] = []
    for i, turn in enumerate(sess.turns):
        if turn.synthetic:
            continue
        if i > 0 and is_pushback(turn.user_prompt):
            rows.append(_row("pushback", sess.id, turn.id, turn.user_prompt))
        reply = "\n".join(turn.assistant_text) or turn.response_preview
        m = _APOLOGY.search(reply)
        if m:
            start = max(0, m.start() - 80)
            rows.append(_row("apology", sess.id, turn.id, reply[start:]))
    return rows


def _row(label: str, sid: str, tid: str, evidence: str, source: str = "auto") -> dict:
    return {
        "label": label,
        "primitive_type": "trace",
        "primitive_id": tid,
        "session_id": sid,
        "evidence": _evidence(evidence),
        "source": source,
    }


def merge_auto(existing: list[dict], found: list[dict], scanned: set[str]) -> list[dict]:
    """Replace automatic labels on the sessions just scanned; keep every hand-set label and the
    automatic labels on sessions this run did not look at."""
    keep = [r for r in existing if r.get("source") != "auto" or r["session_id"] not in scanned]
    return keep + found


def mark(existing: list[dict], label: str, session_ids: list[str], remove: bool) -> list[dict]:
    """Add `label` to whole sessions, or with remove=True drop it from them at every level. A later
    `label` run can find a removed automatic label again."""
    ids = set(session_ids)
    if remove:
        return [r for r in existing if not (r["label"] == label and r["session_id"] in ids)]
    rest = [
        r
        for r in existing
        if not (r["label"] == label and r["primitive_type"] == "session" and r["session_id"] in ids)
    ]
    return rest + [
        {
            "label": label,
            "primitive_type": "session",
            "primitive_id": sid,
            "session_id": sid,
            "evidence": None,
            "source": "manual",
        }
        for sid in session_ids
    ]
