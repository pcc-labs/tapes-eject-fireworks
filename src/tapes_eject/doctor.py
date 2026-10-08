"""`tapes-eject doctor`: is this machine ready to run the demo?"""

from __future__ import annotations

import subprocess
import urllib.request
from typing import Callable

from .config import Config, ping_url

Check = tuple[str, Callable[[], str]]


def run_checks(checks: list[Check]) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for name, fn in checks:
        try:
            lines.append(f"ok    {name}: {fn()}")
        except Exception as e:  # a check's failure is its message, whatever raised it
            ok = False
            lines.append(f"FAIL  {name}: {e}")
    return ok, lines


def _paperctl(*argv: str) -> str:
    proc = subprocess.run(["paperctl", *argv], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"paperctl {argv[0]} failed")
    return proc.stdout


def _paperd() -> str:
    out = _paperctl("status")
    if "healthy" not in out:
        raise RuntimeError("paperd auth is not healthy: run `paperctl login`, then `paperctl init`")
    return "running, auth healthy"


def _paper_org() -> str:
    for line in _paperctl("whoami").splitlines():
        if line.startswith("org_slug:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("`paperctl whoami` printed no org_slug")


def _cassette(base: str) -> str:
    """Optional: only `label` uses a cassette. Labels added in the Paper console need none."""
    try:
        with urllib.request.urlopen(ping_url(base), timeout=5) as resp:
            return f"{base} ({resp.status})"
    except OSError:
        return f"not reachable at {base}; optional, only `label` needs it"


def _fireworks(cfg: Config) -> str:
    from .fireworks import models

    if not cfg.fireworks_api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set; add it to .env")
    return f"key works; {len(models(cfg))} serverless models; training on {cfg.base_model}"


def _tapes(cfg: Config) -> str:
    from .tapes import Tapes

    n = len(Tapes(cfg.tapes_api, cfg.labels_path).sessions(limit=200))
    if not n:
        raise RuntimeError(f"{cfg.tapes_api} has no sessions: import history with tapes first")
    return f"{cfg.tapes_api}, {n}{'+' if n == 200 else ''} sessions"


def _local_labels(cfg: Config) -> str:
    from .tapes import read_labels

    rows = read_labels(cfg.labels_path)
    if not rows:
        return "none yet; run `tapes-eject label`"
    return f"{len(rows)} in {cfg.labels_path}"


def source_checks(cfg: Config) -> list[Check]:
    if cfg.source == "paper":
        return [
            ("paperd", _paperd),
            ("paper org", lambda: cfg.org_slug or _paper_org()),
            ("autolabel cassette", lambda: _cassette(cfg.autolabel_url)),
        ]
    return [("tapes", lambda: _tapes(cfg)), ("labels", lambda: _local_labels(cfg))]


def checks(cfg: Config) -> list[Check]:
    return source_checks(cfg) + [("fireworks", lambda: _fireworks(cfg))]
