"""Client for the autolabel cassette: one run per label, find first, apply only when asked.

POST {base}/run answers 202 with a job; GET {base}/runs/<id> is polled until the state leaves
`running`. The result lists each requested session with the turns the label sits on.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable

from .config import ping_url

Transport = Callable[[str, str, "dict | None"], dict]


class AutolabelError(RuntimeError):
    pass


def http_json(method: str, url: str, body: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise AutolabelError(f"{method} {url}: HTTP {e.code} {e.read()[:300]!r}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AutolabelError(f"{method} {url}: unreachable ({e})") from e


class Autolabel:
    def __init__(
        self,
        base: str,
        transport: Transport = http_json,
        poll_seconds: float = 1.5,
        max_wait_seconds: float = 1800,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base = base.rstrip("/")
        self._t = transport
        self._poll = poll_seconds
        self._max_wait = max_wait_seconds
        self._sleep = sleep

    def ping(self) -> dict:
        return self._t("GET", ping_url(self.base), None)

    def run(
        self,
        label: str,
        session_ids: list[str],
        apply: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict:
        body = {"label": label, "session_ids": list(session_ids), "apply": apply}
        job = self._t("POST", f"{self.base}/run", body)
        waited = 0.0
        while job.get("state", "running") == "running":
            if waited >= self._max_wait:
                raise AutolabelError(f"run {job.get('id')} still running after {waited:.0f}s")
            self._sleep(self._poll)
            waited += self._poll
            job = self._t("GET", f"{self.base}/runs/{job['id']}", None)
            if on_progress and job.get("progress"):
                on_progress(job["progress"])
        if job["state"] != "done":
            raise AutolabelError(f"run {job.get('id')} {job['state']}: {job.get('error', '')}")
        return job["result"]

    def matched_sessions(
        self, label: str, session_ids: list[str], chunk: int = 25
    ) -> tuple[set[str], set[str]]:
        """(matched, unknown). A chunk whose run fails, and any session the cassette could not
        read, is unknown rather than failing the whole call. A `reason` means the label cannot
        run at all (e.g. needs_judge) and raises."""
        hit: set[str] = set()
        unknown: set[str] = set()
        for i in range(0, len(session_ids), chunk):
            part = session_ids[i : i + chunk]
            try:
                res = self.run(label, part)
            except AutolabelError:
                unknown |= set(part)
                continue
            if res.get("reason"):
                raise AutolabelError(f"{label}: {res['reason']}")
            unknown |= set(res.get("missing") or [])
            hit |= {s["session_id"] for s in res["sessions"] if s["matched"]}
        return hit, unknown

    def turn_evidence(
        self, label: str, session_ids: list[str], chunk: int = 25
    ) -> tuple[dict[str, str], str | None]:
        evidence: dict[str, str] = {}
        reason: str | None = None
        for i in range(0, len(session_ids), chunk):
            res = self.run(label, session_ids[i : i + chunk])
            reason = reason or res.get("reason")
            for s in res["sessions"]:
                for turn in s.get("turns") or []:
                    if turn.get("turn_id") and turn.get("evidence"):
                        evidence.setdefault(turn["turn_id"], turn["evidence"])
        return evidence, reason
