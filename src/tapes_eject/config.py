"""Settings, read from the environment. The CLI loads `.env` first."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

NEGATIVE_LABELS = frozenset(
    {"pushback", "apology", "missing-knowledge", "model-error", "regression"}
)
CORRECTION_LABELS = ("pushback", "observation", "missing-knowledge")
GOLDEN = "golden"
REGRESSION = "regression"
NO_OUTCOME = "no-outcome"

DEFAULT_AUTOLABEL_URL = "http://127.0.0.1:9996/v1/cassettes/autolabel"
DEFAULT_TAPES_API = "http://127.0.0.1:18081"
SOURCES = ("tapes", "paper")
# Sessions whose rollup shows more output tokens than this are never exported. Turn count
# alone misses a session of three turns whose tool output runs to hundreds of megabytes, and
# exporting one of those can overload Paper's export service.
DEFAULT_MAX_OUTPUT_TOKENS = 400_000


# Fireworks serverless training is LoRA-only on a shared pool. The cookbook's verified serverless
# model; any model the catalog lists with Availability "Serverless" works, with its HF tokenizer.
DEFAULT_BASE_MODEL = "accounts/fireworks/models/kimi-k3"
DEFAULT_TOKENIZER = "moonshotai/Kimi-K3"
FIREWORKS_API = "https://api.fireworks.ai"


@dataclass(frozen=True)
class Config:
    fireworks_api_key: str | None = None
    base_model: str = DEFAULT_BASE_MODEL
    tokenizer_model: str = DEFAULT_TOKENIZER
    judge_model: str = DEFAULT_BASE_MODEL  # serverless per-token chat model that scores answers
    fireworks_api: str = FIREWORKS_API
    source: str = "tapes"  # tapes: a local tapes stack; paper: a Paper org through paperctl
    tapes_api: str = DEFAULT_TAPES_API
    autolabel_url: str = DEFAULT_AUTOLABEL_URL
    org_slug: str | None = None
    sample_sessions: int = 200
    max_turns: int = 150
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    export_pause: float = 1.0  # seconds between export requests; Paper's export dislikes bursts
    skip_sessions: frozenset[str] = frozenset()  # ids whose export takes Paper's service down
    data_dir: Path = Path("data")

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "local_labels.jsonl"

    def require_key(self) -> str:
        if not self.fireworks_api_key:
            raise SystemExit("missing FIREWORKS_API_KEY (copy .env.example to .env)")
        return self.fireworks_api_key


def load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=value lines into the environment. Variables already set win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load(env: dict[str, str] | None = None) -> Config:
    """Acts 1 and 2 need no Fireworks settings; train, score, and ask call require_key()."""
    env = dict(os.environ) if env is None else env
    return Config(
        fireworks_api_key=env.get("FIREWORKS_API_KEY") or None,
        base_model=env.get("TAPES_EJECT_BASE_MODEL") or DEFAULT_BASE_MODEL,
        tokenizer_model=env.get("TAPES_EJECT_TOKENIZER") or DEFAULT_TOKENIZER,
        judge_model=env.get("TAPES_EJECT_JUDGE_MODEL") or DEFAULT_BASE_MODEL,
        fireworks_api=(env.get("FIREWORKS_BASE_URL") or FIREWORKS_API).rstrip("/"),
        source=_source(env.get("TAPES_EJECT_SOURCE")),
        tapes_api=(env.get("TAPES_API") or DEFAULT_TAPES_API).rstrip("/"),
        autolabel_url=(env.get("AUTOLABEL_URL") or DEFAULT_AUTOLABEL_URL).rstrip("/"),
        org_slug=env.get("PAPER_ORG_SLUG") or None,
        sample_sessions=int(env.get("TAPES_EJECT_SAMPLE_SESSIONS") or 200),
        max_turns=int(env.get("TAPES_EJECT_MAX_TURNS") or 150),
        max_output_tokens=int(
            env.get("TAPES_EJECT_MAX_OUTPUT_TOKENS") or DEFAULT_MAX_OUTPUT_TOKENS
        ),
        export_pause=float(env.get("TAPES_EJECT_EXPORT_PAUSE") or 1.0),
        skip_sessions=frozenset(
            s.strip() for s in (env.get("TAPES_EJECT_SKIP_SESSIONS") or "").split(",") if s.strip()
        ),
    )


def _source(value: str | None) -> str:
    value = (value or "tapes").strip().lower()
    if value not in SOURCES:
        raise SystemExit(f"TAPES_EJECT_SOURCE must be one of {SOURCES}, not {value!r}")
    return value


def ping_url(base: str) -> str:
    """The cassette answers /ping at its host root, whatever its route prefix."""
    parts = urlsplit(base)
    return f"{parts.scheme}://{parts.netloc}/ping"
