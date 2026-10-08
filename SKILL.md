---
name: tapes-eject-fireworks
description: Use when the user wants to fine-tune a model on their own Codex or Claude Code sessions with Fireworks, find the turns where they corrected their agent, or set up, debug, or run tapes-eject-fireworks. Covers importing history with tapes, labeling pushback, golden, and regression sessions, exporting a training set, and running a LoRA fine-tune on Fireworks serverless training.
---

# Fine-tune on a person's agent history with Fireworks

`tapes-eject` reads Codex and Claude Code sessions from a local tapes stack,
labels the turns where the person corrected the agent, and turns the rest
into a training set. `train/fireworks_sft.py` trains a LoRA adapter on
Fireworks serverless training and scores the base model against the tuned
one on those corrections.

Work from the root of this repo. Run every command with `uv run`.

## Order of work

Do these in order. Each step needs the one before it.

1. **History in tapes.** `tapes-skills-demo check`, then `tapes-skills-demo --ollama`.
2. **Repo ready.** `uv sync`, `cp .env.example .env`, `uv run tapes-eject doctor`.
3. **Labels.** `uv run tapes-eject label`, then `mark` by hand.
4. **Data.** `uv run tapes-eject export`, `count`, `prepare`.
5. **Training.** Needs `FIREWORKS_API_KEY`. Smoke run first.

## 1. History in tapes

```bash
go install github.com/pcc-labs/tapes-test/cmd/tapes-skills-demo@main
tapes-skills-demo check --ollama
```

Install from `@main`. The tagged release and the curl installer read Codex
only, so a Claude Code user gets no sessions from them. Confirm `check`
prints a `claude code history` line when the person uses Claude Code. If it
does not, the installed binary is stale: run the `go install` line again.

On any `FAIL` line, stop and tell the user what it says. Docker not running
and a missing key are theirs to fix.

```bash
tapes-skills-demo --ollama                  # last 30 days; --since-days 0 for everything
```

This takes several minutes. Do not time it out. It ends by asking which
skills to write; that part is optional here, so tell the user to answer
`none`. With no terminal attached it writes them all, which is slow with
Ollama but harmless. The import is done once it prints `4/5 deriving`.

## 2. Repo ready

```bash
uv sync
cp .env.example .env                        # only if .env does not exist yet
uv run tapes-eject doctor
```

`doctor` prints one line per check:

| Line | Meaning |
|---|---|
| `tapes: ... N sessions` | The stack answers. `FAIL` here means Docker stopped or nothing was imported. |
| `labels: none yet` | Expected before step 3. Not a failure. |
| `fireworks: FAIL ... not set` | Expected until the user adds a key. Steps 3 and 4 still work. |

## 3. Labels

```bash
uv run tapes-eject label                    # 200 newest sessions
uv run tapes-eject label pushback --show 20
```

`label` matches patterns. It finds `pushback` (the person correcting the
agent) and `apology` (the agent backtracking), and prints the evidence for
each. **Read the evidence with the user.** Pattern matches include false
positives such as "dont send it yet" and plain "no".

Fix labels and add the ones only the person can judge:

```bash
uv run tapes-eject mark pushback <session-id> --remove     # drop a false positive
uv run tapes-eject mark golden <session-id> ...            # sessions worth learning from
uv run tapes-eject mark regression <session-id>            # sessions that went wrong
```

Labels live in `data/local_labels.jsonl`. A later `label` run replaces only
its own labels on the sessions it scans and never touches `mark`'s.
Never mark `golden` or `regression` on the user's behalf without asking:
those are their judgment.

## 4. Data

```bash
uv run tapes-eject export
uv run tapes-eject count
uv run tapes-eject prepare
```

Read `count` with the user before training:

- `training_examples` under 20: training refuses to start. Import more
  history or mark more `golden` sessions.
- `eval_cases` under about 10: base-vs-tuned scores will not mean much. Say so.

Text is redacted for common secret shapes, but redaction is pattern-based.
Before step 5, tell the user that `data/training.jsonl` and the eval prompts
are sent to Fireworks, and suggest they skim `data/training.jsonl`.

## 5. Training

Needs `FIREWORKS_API_KEY` in `.env`. Training costs money, so confirm with
the user before each run. Always run the smoke test first:

```bash
uv sync --group train
uv run --group train python train/fireworks_sft.py --max-steps 2 --limit 3 --no-promote
```

Only after it finishes cleanly, and the user agrees:

```bash
uv run --group train python train/fireworks_sft.py
```

Results land in `data/runs/<run-id>/summary.json`. Report the base and tuned
pass rates and how many cases they cover. With few cases, say the comparison
is weak rather than calling the tuned model better.

The run promotes the adapter and prints a `firectl deployment create`
command. Do not run it unless the user asks: on-demand deployments bill
until deleted.

## When something fails

| It says | Do this |
|---|---|
| `could not reach tapes` | Start Docker, then `tapes-skills-demo --ollama` again. It re-imports nothing. |
| `has no sessions` | `tapes-skills-demo check`; pass `--claude-root` or `--codex-root` if the history lives elsewhere. |
| No Claude Code sessions | The binary is the old release. `go install ...@main` and import again. |
| `only N training examples` | Import more history or `mark` more `golden` sessions, then `export` and `prepare`. |
| `refusing a partial export` | Run `export` again. Pass `--force` to `prepare` only if the user accepts the gaps. |
| Out-of-capacity from Fireworks | The serverless pool is full. Retry later. |

## Cleanup

`tapes-skills-demo down` deletes the local stack and its data, never the
history it read. `data/` holds the exported sessions; it is git-ignored.
