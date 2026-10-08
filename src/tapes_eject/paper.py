"""Read Paper through paperctl and paperd's local proxy. paperd supplies auth and routing; this
module holds no token.

Session records come from `GET /v1/sessions/{id}/traces` through the proxy paperd runs on
localhost, which holds up better under many back-to-back exports than `paperctl sessions
export`. paperctl is the fallback when the daemon reports no proxy.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from typing import Callable

Runner = Callable[[list[str], int], str]
Fetcher = Callable[[str, int], bytes]  # GET url -> body; raises PaperError


class PaperError(RuntimeError):
    pass


def paperctl(argv: list[str], timeout: int) -> str:
    try:
        proc = subprocess.run(
            ["paperctl", *argv], capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise PaperError(f"paperctl {' '.join(argv)} timed out after {timeout}s") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[:500]
        raise PaperError(f"paperctl {' '.join(argv)} exited {proc.returncode}: {detail}")
    return proc.stdout


def http_get(url: str, timeout: int) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return b""
        raise PaperError(f"GET {url} failed: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise PaperError(f"GET {url} could not reach paperd's proxy: {e}") from e


def proxy_from_status(status_text: str) -> str | None:
    """The `proxy:` line of `paperctl status`, as a base URL."""
    m = re.search(r"^proxy:\s+(\S+)", status_text, re.M)
    return f"http://{m.group(1)}" if m else None


class Paper:
    def __init__(
        self,
        run: Runner = paperctl,
        org_slug: str | None = None,
        proxy: str | None = None,
        fetch: Fetcher = http_get,
    ):
        self._run = run
        self._org = org_slug
        self._proxy = proxy.rstrip("/") if proxy else None
        self._proxy_checked = proxy is not None
        self._fetch = fetch

    def proxy(self) -> str | None:
        """paperd's local proxy, found once from `paperctl status`; None if the daemon has none."""
        if not self._proxy_checked:
            self._proxy_checked = True
            try:
                self._proxy = proxy_from_status(self._run(["status"], 30))
            except PaperError:
                self._proxy = None
        return self._proxy

    def _call(self, argv: list[str], timeout: int = 120) -> str:
        prefix = ["--org-slug", self._org] if self._org else []
        return self._run([*prefix, *argv], timeout)

    def labels(self) -> list[dict]:
        return json.loads(self._call(["cassettes", "labels", "list-labels"]))["labels"]

    def attachments(self, label_id: str, primitive_type: str) -> list[dict]:
        out: list[dict] = []
        cursor: str | None = None
        while True:
            argv = ["cassettes", "labels", "list-label-attachments", label_id]
            if cursor:
                argv += ["--cursor", cursor]
            argv += ["--primitive-type", primitive_type, "--limit", "200"]
            page = json.loads(self._call(argv))
            out.extend(page.get("attachments") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                return out

    def sessions(
        self, label: str | None = None, since: str | None = None, limit: int = 200
    ) -> list[dict]:
        """Session list items, newest first, paged until `limit`."""
        out: list[dict] = []
        cursor: str | None = None
        while len(out) < limit:
            argv = ["sessions", "list", "--json", "--limit", str(min(200, limit - len(out)))]
            if cursor:
                argv += ["--cursor", cursor]
            if since:
                argv += ["--since", since]
            if label:
                argv += ["--label", label]
            page = json.loads(self._call(argv))
            items = page.get("items") or []
            out.extend(items)
            cursor = page.get("next_cursor")
            if not cursor or not items:
                break
        return out[:limit]

    def export_session(self, session_id: str, timeout: int = 300) -> dict | None:
        """The session's full record: {schema, session, traces: [{trace, spans}], links}. From
        paperd's proxy when there is one, else paperctl's export (never with
        --detail: `traces` strips spans). The proxy uses the daemon's active org."""
        proxy = self.proxy()
        if proxy:
            body = self._fetch(f"{proxy}/v1/sessions/{session_id}/traces", timeout)
            if not body:
                return None
            try:
                return json.loads(body)
            except ValueError as e:
                raise PaperError(
                    f"sessions traces {session_id} truncated at {len(body)} bytes: {e}"
                ) from e
        text = self._call(["sessions", "export", session_id], timeout)
        for line in text.splitlines():
            if line.strip():
                try:
                    return json.loads(line)
                except ValueError as e:
                    # Paper's export service cuts the stream when it goes down mid-response;
                    # the record arrives truncated. Report it as such, never cache it.
                    raise PaperError(
                        f"sessions export {session_id} truncated at {len(line)} bytes: {e}"
                    ) from e
        return None
