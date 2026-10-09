# tapes-eject-fireworks

Fine-tune a model on your own Codex and Claude Code sessions with [Fireworks serverless training](https://docs.fireworks.ai/fine-tuning/training-api/serverless).

[tapes](https://github.com/pcc-labs/tapes-test) imports your agent history into a database on your laptop. This repo reads it from there, finds the turns where you corrected the agent, turns the rest into a training set, trains a LoRA adapter on Fireworks, and scores the base model against the tuned one on your corrections.

Everything up to training runs locally. The only data that leaves your machine is the training set and the eval prompts sent to Fireworks. No GPU is needed: Fireworks trains on a shared pool and bills per token.

## How it works

```
~/.codex/sessions, ~/.claude/projects
  └─ tapes-skills-demo        import history into local tapes (Docker)
  └─ tapes-eject label        find pushback and apology turns
  └─ tapes-eject mark         mark sessions golden or regression by hand
  └─ tapes-eject export       sessions, turns, labels       -> data/
  └─ tapes-eject prepare      training.jsonl, eval_cases.jsonl
  └─ train/fireworks_sft.py   LoRA SFT on Fireworks -> score base vs tuned -> promote
```

Labels decide the data:

- **`pushback`:** a turn where you corrected the agent ("no, ...", "don't ...", "you deleted ..."). Found by `label`.
- **`apology`:** a turn where the agent backtracked ("you're right", "my mistake"). Found by `label`.
- **`golden`:** a session worth learning from. Set by hand with `mark`.
- **`regression`:** a session that went wrong. Set by hand with `mark`.

Sessions with no correction or failure label become training examples, along with four in five `golden` sessions. Every corrected turn, every `regression` session, and the remaining `golden` sessions become eval cases. A session is never in both sets. An eval case asks the model to answer the conversation up to your correction, and a judge model checks whether the answer already does what you had to ask for.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Docker, running
- Codex or Claude Code history on this machine
- A [Fireworks](https://fireworks.ai) account and API key, for training only

## Setup

### 1. Import your history into tapes

Install [tapes-test](https://github.com/pcc-labs/tapes-test) and run it once. It starts a local tapes stack in Docker on port 18081 and imports the last 30 days of Codex and Claude Code sessions.

```bash
curl -fsSL https://raw.githubusercontent.com/pcc-labs/tapes-test/main/install.sh | sh
tapes-skills-demo check
tapes-skills-demo --ollama        # or export OPENAI_API_KEY and drop --ollama
tapes-skills-demo sessions        # what was imported
```

`check` lists Codex and Claude Code history separately; you need one of the two. With Go installed, `go install github.com/pcc-labs/tapes-test/cmd/tapes-skills-demo@latest` works too.

`tapes-skills-demo` also suggests skills written from your history. That part is optional here: answer `none` when it asks which to write. Use `--since-days 0` to import everything, and run it again later to pick up new sessions.

### 2. Install this repo

```bash
git clone https://github.com/pcc-labs/tapes-eject-fireworks
cd tapes-eject-fireworks
cp .env.example .env              # set FIREWORKS_API_KEY when you are ready to train
uv sync
uv run tapes-eject doctor
```

`doctor` checks that tapes answers on `TAPES_API`, how many labels you have, and that the Fireworks key works.

## Usage

### 1. Label your sessions

```bash
uv run tapes-eject label                 # scan the 200 newest sessions
uv run tapes-eject label pushback --show 20
```

`label` matches patterns, so it is fast, free, and sometimes wrong. It prints the evidence for each label: read it. Labels are kept in `data/local_labels.jsonl`. Re-running `label` replaces its own labels on the sessions it scans and never touches the ones you set by hand.

```bash
uv run tapes-eject mark golden <session-id> <session-id>     # sessions worth learning from
uv run tapes-eject mark regression <session-id>              # sessions that went wrong
uv run tapes-eject mark pushback <session-id> --remove       # drop a wrong label
```

Session ids come from `tapes-skills-demo sessions` or from `label`'s output.

### 2. Export and inspect

```bash
uv run tapes-eject export      # sessions, turns, and labels into data/
uv run tapes-eject count       # how many training examples and eval cases the labels select
uv run tapes-eject report                              # each label's rate by model
uv run tapes-eject report --by project --label pushback
uv run tapes-eject report --by week
uv run tapes-eject prepare     # data/training.jsonl, eval_cases.jsonl, corrections.jsonl
```

`export` caches sessions in `data/cache/` and skips sessions over the size caps in `.env`. Text is redacted for common secret shapes before it is written. Redaction is pattern-based, so read `data/training.jsonl` before you send it anywhere.

`prepare` refuses an export where some sessions failed. Re-run `export`, or pass `--force`.

### 3. Train on Fireworks

```bash
uv sync --group train
uv run --group train python train/fireworks_sft.py --max-steps 2 --limit 3 --no-promote   # smoke test
uv run --group train python train/fireworks_sft.py                                        # full run
```

The script:

1. Opens a serverless training session and a LoRA training client on the base model.
2. Saves a checkpoint of the untrained adapter, so the base model is sampled the same way as the tuned one.
3. Runs cross-entropy SFT for two epochs (or `--max-steps`), printing the loss each step.
4. Samples base and tuned on each eval case, and has `TAPES_EJECT_JUDGE_MODEL` grade both answers.
5. Promotes the final adapter to a Fireworks model and prints the deploy command.

Output lands in `data/runs/<run-id>/`: `metrics.jsonl`, `scores.jsonl`, and `summary.json`.

Useful flags: `--batch-size`, `--learning-rate`, `--lora-rank`, `--epochs`, `--limit` (score only the first N eval cases), and `--no-promote`. Training needs at least 20 examples; pass `--min-examples` to change that.

### 4. Serve the tuned model

Serverless per-token serving does not support trained LoRAs yet, so deploy it on demand:

```bash
firectl deployment create "accounts/<account>/models/<model-id>" --deployment-shape default
uv run tapes-eject ask --model "accounts/<account>/models/<model-id>" "Add a --dry-run flag to the export command"
```

On-demand deployments bill while they run. Delete the deployment when you are done.

## Configuration

All settings live in `.env`; see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `TAPES_API` | `http://127.0.0.1:18081` | The tapes API to read sessions from |
| `FIREWORKS_API_KEY` | | Required for training and `ask` |
| `TAPES_EJECT_BASE_MODEL` | `accounts/fireworks/models/kimi-k3` | Any model listed with Availability "Serverless" |
| `TAPES_EJECT_TOKENIZER` | `moonshotai/Kimi-K3` | Hugging Face tokenizer matching the base model |
| `TAPES_EJECT_JUDGE_MODEL` | same as base | Chat model that grades eval answers |
| `TAPES_EJECT_SAMPLE_SESSIONS` | `200` | Unlabeled recent sessions to export alongside labeled ones |
| `TAPES_EJECT_MAX_TURNS` | `150` | Skip sessions longer than this |
| `TAPES_EJECT_MAX_OUTPUT_TOKENS` | `400000` | Skip sessions with more output tokens than this |
| `TAPES_EJECT_SKIP_SESSIONS` | | Comma-separated session ids never to export |
| `TAPES_EJECT_SOURCE` | `tapes` | `paper` reads a [Paper](https://papercompute.com) org through `paperctl` instead, with labels set in the Paper console |

## Notes on Fireworks serverless training

- Serverless training is LoRA only. Full-parameter training needs dedicated training.
- Billing is per token on three meters: prompt tokens, sampled tokens, and trained tokens.
- A session's checkpoints can only be promoted while the session is open, so the script trains, scores, and promotes in one process.
- Checkpoint names must be 17 characters or fewer. The script checks this before training starts.

## Troubleshooting

- **`doctor` says tapes could not be reached.** Start Docker, then run `tapes-skills-demo` again. It restarts the stack and only imports what is new.
- **`doctor` says tapes has no sessions.** Run `tapes-skills-demo check` to see which history it finds. Pass `--codex-root` or `--claude-root` if yours lives elsewhere.
- **"only N training examples".** Import more history (`tapes-skills-demo --since-days 0`), or `mark` more sessions `golden`, then run `export` and `prepare` again.
- **Very few eval cases.** Eval cases come from corrected turns and `golden`/`regression` sessions. With few of those, base-vs-tuned scores won't mean much.
- **Out-of-capacity error from Fireworks.** The serverless pool is full. Retry later.
- **The tokenizer asks for `trust_remote_code`.** Kimi's tokenizer ships custom code. The script loads it through the Fireworks cookbook loader, which allows it.

## Using it with an agent

This repo ships a [SKILL.md](SKILL.md) that walks an agent through setup, labeling, and training, including where to stop and ask you. Point Claude Code or Codex at it:

> Read SKILL.md in this repo and follow it against my agent history.

## Development

```bash
uv run pytest
uv run ruff check .
```

The tests need no tapes stack, no `train` group, and no Fireworks key.

## License

MIT
