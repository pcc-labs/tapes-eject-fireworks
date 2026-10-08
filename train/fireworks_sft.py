"""Act 3 on Fireworks serverless training: LoRA SFT on the examples Paper's labels selected,
then base and tuned scored on the eval cases, then the adapter promoted to a Fireworks model.

There is no job to submit and no GPU to provision. This process connects to Fireworks' shared
pooled trainer and runs the loop itself: forward_backward and optim_step happen on remote GPUs,
and billing is per token (prefill, sample, train). Everything happens inside one training
session, because a session's checkpoints can only be listed or promoted while it is alive.

    uv sync --group train
    uv run --group train python train/fireworks_sft.py --max-steps 2 --limit 3 --no-promote  # smoke
    uv run --group train python train/fireworks_sft.py                                       # full

The base model is scored through a sampler snapshot saved before the first optimizer step, so
base and tuned are rendered, sampled, and judged exactly the same way.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tapes_eject import config, curate  # noqa: E402
from tapes_eject import fireworks as fw  # noqa: E402


def _mean_loss(fb) -> float | None:
    metrics = getattr(fb, "metrics", None) or {}
    loss_sum = metrics.get("loss:sum")
    tokens = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
    return None if loss_sum is None else float(loss_sum) / max(float(tokens), 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/training.jsonl"))
    p.add_argument("--eval", type=Path, default=Path("data/eval_cases.jsonl"))
    p.add_argument("--max-steps", type=int, default=0, help="0 trains --epochs passes")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--max-sample-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="score only the first N eval cases")
    p.add_argument("--min-examples", type=int, default=20)
    p.add_argument("--no-promote", action="store_false", dest="promote")
    p.add_argument("--output-model-id", default="tapes-eject-agent")
    args = p.parse_args()

    config.load_dotenv()
    cfg = config.load()
    api_key = cfg.require_key()
    if not args.data.exists():
        raise SystemExit(f"no {args.data}: run `tapes-eject prepare` first")
    rows = curate.read_jsonl(args.data)
    if len(rows) < args.min_examples:
        raise SystemExit(
            f"only {len(rows)} training examples (need {args.min_examples}); mark more good "
            "sessions `golden` in Paper, then run export and prepare again"
        )
    cases = curate.read_jsonl(args.eval) if args.eval.exists() else []
    if args.limit:
        cases = cases[: args.limit]

    import tinker
    from fireworks.training.sdk import (
        FiretitanSamplingParams,
        FiretitanServiceClient,
        FireworksClient,
    )
    from training.renderer import get_text_content
    from training.utils.supervised import (
        build_renderer,
        render_messages_to_datum,
        resolve_renderer_name,
    )
    from training.utils.tokenizers import load_tokenizer

    tokenizer = load_tokenizer(cfg.tokenizer_model)  # Kimi ships custom tokenizer code
    renderer = build_renderer(
        tokenizer, cfg.tokenizer_model, resolve_renderer_name(cfg.tokenizer_model)
    )

    datums, dropped = [], 0
    for row in rows:
        try:
            rendered = render_messages_to_datum(
                row["messages"], renderer=renderer, max_seq_len=args.max_seq_len
            )
        except ValueError:
            dropped += 1
            continue
        if len(rendered.token_ids) > args.max_seq_len or not any(rendered.token_weights):
            dropped += 1
            continue
        datums.append(rendered.datum)
    steps = fw.plan_steps(len(datums), args.batch_size, args.epochs, args.max_steps)

    service = FiretitanServiceClient(
        api_key=api_key, base_url=f"{cfg.fireworks_api}/training/v1/serverless"
    )
    client = service.create_lora_training_client(base_model=cfg.base_model, rank=args.lora_rank)
    session, run_id = service.training_session_name, client.run_id
    run_dir = cfg.data_dir / "runs" / (run_id or f"run-{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"session {session} run {run_id}\n{cfg.base_model}: {len(datums)} examples "
        f"({dropped} over {args.max_seq_len} tokens or untrainable, left out), {steps} steps "
        f"of {args.batch_size}, LoRA rank {args.lora_rank}, {len(cases)} eval cases",
        flush=True,
    )

    try:
        base = client.save_weights_for_sampler(fw.check_checkpoint_name("base")).result().path
        adam = tinker.AdamParams(
            learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0
        )
        with (run_dir / "metrics.jsonl").open("w") as log:
            for step in range(steps):
                t0 = time.time()
                fb = client.forward_backward(
                    fw.batch(datums, step, args.batch_size), "cross_entropy"
                ).result()
                client.optim_step(adam).result()
                loss = _mean_loss(fb)
                log.write(json.dumps({"step": step, "loss": loss}) + "\n")
                shown = "n/a" if loss is None or not math.isfinite(loss) else f"{loss:.4f}"
                print(f"step {step:03d} loss={shown} ({time.time() - t0:.1f}s)", flush=True)
        client.save_state(fw.check_checkpoint_name("final-state")).result(timeout=900)
        tuned = client.save_weights_for_sampler(fw.check_checkpoint_name("final")).result().path
        print(f"tuned sampler checkpoint: {tuned}", flush=True)

        if cases:
            score(
                cfg,
                service,
                tokenizer,
                renderer,
                base,
                tuned,
                cases,
                args,
                run_dir,
                FiretitanSamplingParams,
                get_text_content,
            )
        if args.promote:
            promote(cfg, api_key, session, run_id, args.output_model_id, FireworksClient)
    finally:
        service.close()
    return 0


def _sample(service, snapshot, tokenizer, renderer, cases, params, get_text_content) -> list[str]:
    sampler = service.create_sampling_client(model_path=snapshot, tokenizer=tokenizer)
    try:
        futures = [
            sampler.sample(
                prompt=renderer.build_generation_prompt(c["inputs"]["messages"]),
                num_samples=1,
                sampling_params=params,
            )
            for c in cases
        ]
        out = []
        for fut in futures:
            seq = fut.result(timeout=600).sequences[0]
            out.append(get_text_content(renderer.parse_response(list(seq.tokens))[0]))
        return out
    finally:
        sampler.close()


def score(cfg, service, tokenizer, renderer, base, tuned, cases, args, run_dir, Params, text):
    params = Params(
        max_tokens=args.max_sample_tokens, temperature=0.0, stop=renderer.get_stop_sequences()
    )
    print(f"sampling base and tuned on {len(cases)} eval cases...", flush=True)
    answers = {
        "base": _sample(service, base, tokenizer, renderer, cases, params, text),
        "tuned": _sample(service, tuned, tokenizer, renderer, cases, params, text),
    }
    scores = []
    for i, case in enumerate(cases):
        msgs, rules = case["inputs"]["messages"], case["expectations"]["guidelines"]
        row = {"case": i, "guidelines": rules}
        for which in ("base", "tuned"):
            passed, why = fw.judge(cfg, msgs, answers[which][i], rules)
            row.update({which: answers[which][i], f"{which}_pass": passed, f"{which}_why": why})
        scores.append(row)
    with (run_dir / "scores.jsonl").open("w") as fh:
        fh.writelines(json.dumps(s) + "\n" for s in scores)
    summary = fw.summarize(scores)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for which in ("base", "tuned"):
        s = summary[which]
        print(f"{which:>6}: {s['passed']}/{s['of']} cases pass ({s['rate']:.0%})")
    if summary["unjudged"]["cases"]:
        print(f"{summary['unjudged']['cases']} cases left out: the judge gave no verdict")
    print(f"scores: {run_dir / 'scores.jsonl'}")


def promote(cfg, api_key, session, run_id, output_model_id, FireworksClient) -> None:
    cp = FireworksClient(api_key=api_key, base_url=cfg.fireworks_api)
    try:
        match = fw.find_promotable(cp.list_training_session_checkpoints(session), "final", run_id)
        if match is None:
            print("promote skipped: the final checkpoint is not listed as promotable yet")
            return
        model = cp.promote_session_checkpoint(
            name=match["name"],
            output_model_id=fw.model_id(output_model_id, run_id),
            base_model=cfg.base_model,
        )
        name = model.get("name") if isinstance(model, dict) else str(model)
        print(
            f"promoted: {name}\n"
            "serving a LoRA needs an on-demand deployment for now (serverless LoRA is coming):\n"
            f'  firectl deployment create "{name}" --deployment-shape default\n'
            f'  uv run tapes-eject ask --model "{name}" "<prompt>"'
        )
    finally:
        cp.close()


if __name__ == "__main__":
    sys.exit(main())
