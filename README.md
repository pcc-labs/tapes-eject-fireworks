# tapes-eject-fireworks

Fine-tune a model on your own coding-agent sessions with [Fireworks serverless training](https://docs.fireworks.ai/fine-tuning/training-api/serverless).

Sessions captured by [Paper](https://papercompute.com) carry labels: `pushback` where an engineer corrected the agent, `apology` where the agent backtracked, `golden` for sessions worth learning from, `regression` for known failures. This repo turns those labels into a training set and an eval set, trains a LoRA adapter on Fireworks, and scores the base model against the tuned one on the turns engineers had to correct.

No GPU or cluster is needed. Fireworks runs the training on a shared pool and bills per token.

## How it works

```
Paper sessions + labels
  └─ tapes-eject export     sessions, turns, labels  -> data/
  └─ tapes-eject prepare    training.jsonl, eval_cases.jsonl
  └─ train/fireworks_sft.py LoRA SFT on Fireworks -> score base vs tuned -> promote
```

Labels decide the data:

- **Training examples:** sessions that produced an outcome and carry no correction or failure label (`pushback`, `apology`, `missing-knowledge`, `model-error`, `observation`, `regression`). Four in five `golden` sessions are training examples too.
- **Eval cases:** every turn an engineer corrected, every `regression` session, and the remaining one in five `golden` sessions. A session is never in both sets.

An eval case asks the model to answer the conversation up to the corrected turn. A judge model then checks whether the answer already does what the engineer had to ask for.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A Paper account with `paperctl` installed and logged in
- A [Fireworks](https://fireworks.ai) account and API key

## Setup

```bash
git clone https://github.com/pcc-labs/tapes-eject-fireworks
cd tapes-eject-fireworks
cp .env.example .env          # set FIREWORKS_API_KEY
uv sync

paperctl login
paperctl status               # expect "auth: healthy"
uv run tapes-eject doctor
```

`doctor` checks that paperd is running, which Paper org you are reading, and that the Fireworks key works. Every line except the optional autolabel line should say `ok`.

## Usage

### 1. Label sessions

Add labels in the Paper console: mark a few good sessions `golden`, and failures `regression`. Corrections (`pushback`) and backtracks (`apology`) are the most useful labels for evaluation.

If you run an autolabel service, `label` finds a label across recent sessions and can write it back to Paper:

```bash
uv run tapes-eject label apology --sessions 500            # report matches only
uv run tapes-eject label apology --sessions 500 --apply    # write them to Paper
```

Set `AUTOLABEL_URL` in `.env` to point at it.

### 2. Export and inspect

```bash
uv run tapes-eject export      # sessions, turns, and labels into data/
uv run tapes-eject count       # how many training examples and eval cases the labels select
uv run tapes-eject report                              # each label's rate by model
uv run tapes-eject report --by project --label pushback
uv run tapes-eject report --by week
uv run tapes-eject prepare     # writes data/training.jsonl and data/eval_cases.jsonl
```

`export` caches every session in `data/cache/`, so a re-run only fetches sessions that changed. It skips sessions over the size caps in `.env`. Text is redacted for common secret shapes before it is written. Redaction is pattern-based, so check `data/` before you share it.

`prepare` refuses an export where some sessions failed, because training on it would silently drop data. Re-run `export`, or pass `--force`.

### 3. Train on Fireworks

```bash
uv sync --group train
uv run --group train python train/fireworks_sft.py --max-steps 2 --limit 3 --no-promote   # smoke test
uv run --group train python train/fireworks_sft.py                                        # full run
```

The script:

1. Opens a serverless training session and a LoRA training client on the base model.
2. Saves a checkpoint of the untrained adapter, so the base model can be sampled the same way as the tuned one.
3. Runs cross-entropy SFT for two epochs (or `--max-steps`), printing the loss each step.
4. Samples base and tuned on each eval case, and has `TAPES_EJECT_JUDGE_MODEL` grade both answers.
5. Promotes the final adapter to a Fireworks model and prints the deploy command.

Output lands in `data/runs/<run-id>/`: `metrics.jsonl`, `scores.jsonl`, and `summary.json`.

Useful flags: `--batch-size`, `--learning-rate`, `--lora-rank`, `--epochs`, `--limit` (score only the first N eval cases), and `--no-promote`.

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
| `FIREWORKS_API_KEY` | | Required for training and `ask` |
| `TAPES_EJECT_BASE_MODEL` | `accounts/fireworks/models/kimi-k3` | Any model listed with Availability "Serverless" |
| `TAPES_EJECT_TOKENIZER` | `moonshotai/Kimi-K3` | Hugging Face tokenizer matching the base model |
| `TAPES_EJECT_JUDGE_MODEL` | same as base | Chat model that grades eval answers |
| `PAPER_ORG_SLUG` | paperctl's active org | Which Paper org to read |
| `TAPES_EJECT_SAMPLE_SESSIONS` | `200` | Unlabeled recent sessions to export alongside labeled ones |
| `TAPES_EJECT_MAX_TURNS` | `150` | Skip sessions longer than this |
| `TAPES_EJECT_MAX_OUTPUT_TOKENS` | `400000` | Skip sessions with more output tokens than this |
| `TAPES_EJECT_SKIP_SESSIONS` | | Comma-separated session ids never to export |

## Notes on Fireworks serverless training

- Serverless training is LoRA only. Full-parameter training needs dedicated training.
- Billing is per token on three meters: prompt tokens, sampled tokens, and trained tokens.
- A session's checkpoints can only be promoted while the session is open, so the script trains, scores, and promotes in one process.
- Checkpoint names must be 17 characters or fewer. The script checks this before training starts.

## Troubleshooting

- **Out-of-capacity error from Fireworks.** The serverless pool is full. Retry later.
- **"only N training examples".** Label more sessions `golden`, then run `export` and `prepare` again.
- **Very few eval cases.** Eval cases come from corrected turns and `golden`/`regression` sessions. With few of those labels, base-vs-tuned scores won't mean much.
- **The tokenizer asks for `trust_remote_code`.** Kimi's tokenizer ships custom code. The script loads it through the Fireworks cookbook loader, which allows it.
- **Export fails with "could not reach".** Paper's export is not answering. Wait and re-run; cached sessions are not fetched again.

## Development

```bash
uv run pytest
uv run ruff check .
```

The tests need neither the `train` group nor a Fireworks key.

## License

MIT
