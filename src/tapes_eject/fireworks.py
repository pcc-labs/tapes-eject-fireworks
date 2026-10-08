"""Fireworks over plain HTTP, plus the pure pieces of the serverless training run.

Nothing here imports the training SDK, so Acts 1 and 2 and the unit tests run without the heavy
`train` dependency group. train/fireworks_sft.py holds the loop that does.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request

from .config import Config

# The trainer composes a promotable checkpoint id as "{run_id}-{name}-{suffix}" against a
# 63-character limit, and only checks it at promote time, after training has finished.
MAX_CHECKPOINT_NAME = 17
MAX_MODEL_ID = 63


def _post(cfg: Config, path: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(
        f"{cfg.fireworks_api}{path}",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {cfg.require_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"fireworks {path}: HTTP {e.code} {e.read().decode()[:300]}") from e


def models(cfg: Config) -> list[str]:
    """Model ids the key can call; doctor uses it to prove the key works."""
    req = urllib.request.Request(
        f"{cfg.fireworks_api}/inference/v1/models",
        headers={"Authorization": f"Bearer {cfg.require_key()}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return [m["id"] for m in json.loads(resp.read()).get("data", [])]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: is FIREWORKS_API_KEY valid?") from e


def chat(
    cfg: Config, model: str, messages: list[dict], max_tokens: int = 1024, temperature: float = 0
) -> str:
    out = _post(
        cfg,
        "/inference/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    return out["choices"][0]["message"]["content"] or ""


JUDGE_SYSTEM = (
    "You grade a coding agent's reply against guidelines written from a real engineer's "
    "session. Answer with one JSON object and nothing else: "
    '{"pass": true|false, "why": "<one sentence>"}. Pass only if the reply meets every guideline.'
)


def judge_messages(conversation: list[dict], answer: str, guidelines: list[str]) -> list[dict]:
    convo = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in conversation)
    rules = "\n".join(f"- {g}" for g in guidelines)
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation so far:\n{convo}\n\nReply to grade:\n{answer}\n\n"
            f"Guidelines:\n{rules}",
        },
    ]


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """(passed, why). None when the judge returned no readable verdict; that case is not
    counted for either model rather than guessed."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            got = json.loads(match.group(0))
            if isinstance(got.get("pass"), bool):
                return got["pass"], str(got.get("why", ""))
        except json.JSONDecodeError:
            pass
    return None, text.strip()[:200]


def judge(cfg: Config, conversation: list[dict], answer: str, guidelines: list[str]):
    return parse_verdict(
        chat(cfg, cfg.judge_model, judge_messages(conversation, answer, guidelines), 300)
    )


def plan_steps(examples: int, batch_size: int, epochs: float, max_steps: int) -> int:
    """max_steps > 0 is a smoke run; 0 trains `epochs` passes over the examples."""
    if max_steps > 0:
        return max_steps
    return max(1, math.ceil(examples * epochs / batch_size))


def batch(rows: list, step: int, size: int) -> list:
    """The rows for one optimizer step, wrapping around the dataset."""
    return [rows[(step * size + i) % len(rows)] for i in range(size)]


def check_checkpoint_name(name: str) -> str:
    if not name or len(name) > MAX_CHECKPOINT_NAME:
        raise ValueError(
            f"checkpoint name {name!r} must be 1-{MAX_CHECKPOINT_NAME} characters; a longer one "
            "only fails at promote time, after training has finished"
        )
    return name


def find_promotable(checkpoints: list[dict], name: str, run_id: str | None) -> dict | None:
    """The listed, promotable checkpoint saved as `name`. The trainer lists it bare or as
    `{run_id}-{name}-{suffix}`; filter on promotable, not on name equality alone."""
    prefixes = [name] + ([f"{run_id}-{name}"] if run_id else [])

    def label(c: dict) -> str:
        return str(c.get("name", "")).rstrip("/").split("/")[-1]

    return next(
        (
            c
            for c in checkpoints
            if c.get("promotable")
            and any(label(c) == p or label(c).startswith(p + "-") for p in prefixes)
        ),
        None,
    )


def model_id(base: str, run_id: str | None) -> str:
    """Unique per run so a second run never collides with an already-promoted model. Model ids
    are lowercase alphanumerics and hyphens, at most 63 characters."""
    base = re.sub(r"[^a-z0-9-]+", "-", base.lower()).strip("-")
    suffix = str(run_id or "").replace("run-", "")[:8]
    if not suffix:
        return base[:MAX_MODEL_ID]
    return f"{base[: MAX_MODEL_ID - len(suffix) - 1]}-{suffix}"


def summarize(scores: list[dict]) -> dict[str, dict]:
    """Pass rate per model over the cases the judge could read for both models."""
    both = [s for s in scores if s["base_pass"] is not None and s["tuned_pass"] is not None]
    out = {}
    for which in ("base", "tuned"):
        passed = sum(1 for s in both if s[f"{which}_pass"])
        out[which] = {
            "passed": passed,
            "of": len(both),
            "rate": round(passed / len(both), 3) if both else 0.0,
        }
    out["unjudged"] = {"cases": len(scores) - len(both)}
    return out
