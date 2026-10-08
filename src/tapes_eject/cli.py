"""tapes-eject: labeled agent sessions to a Fireworks serverless LoRA fine-tune."""

from __future__ import annotations

import argparse
import json
import sys

from . import config, curate, doctor, report
from .autolabel import Autolabel
from .export import _cached_export, _write_jsonl, problems, run_export, too_big, write_export
from .paper import Paper
from .session import parse_session
from .tapes import (
    AUTO_LABELS,
    MANUAL_LABELS,
    Tapes,
    find_labels,
    mark,
    merge_auto,
    read_labels,
    write_labels,
)


def cmd_doctor(cfg: config.Config, args: argparse.Namespace) -> int:
    ok, lines = doctor.run_checks(doctor.checks(cfg))
    print("\n".join(lines))
    if not ok:
        print("\nFix the FAIL lines; README 'Setup' covers each one.")
    return 0 if ok else 1


def label_candidates(
    items: list[dict], cfg: config.Config
) -> tuple[list[str], list[tuple[str, str]]]:
    """(ids to send to the cassette, (id, reason) for each session over the size caps)."""
    ids: list[str] = []
    skipped: list[tuple[str, str]] = []
    for it in items:
        if not ((it.get("rollup") or {}).get("turn_count") or 0):
            continue  # an empty session cannot carry a label; do not export it
        reason = too_big(it, cfg.max_turns, cfg.max_output_tokens, cfg.skip_sessions)
        if reason:
            skipped.append((it["id"], reason))
        else:
            ids.append(it["id"])
    return ids, skipped


def source(cfg: config.Config):
    """Where sessions come from: a local tapes stack (default) or a Paper org."""
    if cfg.source == "paper":
        return Paper(org_slug=cfg.org_slug)
    return Tapes(cfg.tapes_api, cfg.labels_path)


def label_local(cfg: config.Config, args: argparse.Namespace) -> int:
    """Find `pushback` and `apology` turns in the newest local sessions and keep them in
    data/local_labels.jsonl. Hand-set labels are never touched."""
    tapes = Tapes(cfg.tapes_api, cfg.labels_path)
    items = tapes.sessions(limit=args.sessions)
    ids, skipped = label_candidates(items, cfg)
    for sid, reason in skipped:
        print(f"  skip {sid}: {reason}")
    seen = {it["id"]: it.get("last_seen_at") for it in items}
    found: list[dict] = []
    scanned: set[str] = set()
    for i, sid in enumerate(ids, 1):
        print(f"  [{i}/{len(ids)}] {sid}", file=sys.stderr)
        rec, _ = _cached_export(tapes, sid, seen.get(sid), cfg.data_dir / "cache")
        sess = parse_session(rec) if rec else None
        if sess is None:
            continue
        scanned.add(sid)
        found.extend(find_labels(sess))
    write_labels(cfg.labels_path, merge_auto(read_labels(cfg.labels_path), found, scanned))
    for name in AUTO_LABELS:
        if args.name and name != args.name:
            continue
        rows = [r for r in found if r["label"] == name]
        sessions = len({r["session_id"] for r in rows})
        print(f"{name}: {len(rows)} turns in {sessions} of {len(scanned)} sessions")
        for r in rows[: args.show]:
            print(f"  {r['session_id'][:13]}  {r['evidence']}")
    print(f"labels -> {cfg.labels_path} (pattern-matched: read the evidence, fix with `mark`)")
    return 0


def cmd_mark(cfg: config.Config, args: argparse.Namespace) -> int:
    if cfg.source == "paper":
        print("with TAPES_EJECT_SOURCE=paper, set labels in the Paper console")
        return 1
    rows = mark(read_labels(cfg.labels_path), args.label, args.session_ids, args.remove)
    write_labels(cfg.labels_path, rows)
    verb = "removed from" if args.remove else "added to"
    print(f"{args.label} {verb} {len(args.session_ids)} session(s) in {cfg.labels_path}")
    return 0


def cmd_label(cfg: config.Config, args: argparse.Namespace) -> int:
    """Local tapes: find labels by pattern. Paper: find a label through the autolabel cassette,
    and with --apply write it to Paper. Sessions over the size caps are left out either way."""
    if cfg.source != "paper":
        return label_local(cfg, args)
    if not args.name:
        print("with TAPES_EJECT_SOURCE=paper, name the label to find, e.g. `label pushback`")
        return 1
    items = Paper(org_slug=cfg.org_slug).sessions(limit=args.sessions)
    ids, skipped = label_candidates(items, cfg)
    for sid, reason in skipped:
        print(f"  skip {sid}: {reason}")
    if not ids:
        print(f"{args.name}: none of the {len(items)} newest sessions are under the size caps")
        return 1
    client = Autolabel(cfg.autolabel_url)
    res = client.run(args.name, ids, apply=False, on_progress=lambda p: print(f"  {p}"))
    if res.get("reason"):
        print(f"{args.name}: {res['reason']}")
        return 1
    print(f"{args.name} found in {res['matched']} of {len(ids)} sessions")
    if not args.apply or not res["matched"]:
        return 0
    done = client.run(args.name, ids, apply=True, on_progress=lambda p: print(f"  {p}"))
    print(f"{args.name}: {done['created']} new labels, {done['pushed']} pushed to Paper")
    return 0 if not done["push_failed"] else 1


def export_status(report: dict, allow_partial: bool) -> int:
    issues = problems(report)
    for issue in issues:
        print(f"problem: {issue}")
    if issues and not allow_partial:
        print("sync will refuse this export; rerun `export`, or pass --allow-partial to accept it")
        return 1
    return 0


def cmd_export(cfg: config.Config, args: argparse.Namespace) -> int:
    ex = run_export(source(cfg), cfg, cache_dir=cfg.data_dir / "cache")
    write_export(ex, cfg.data_dir)
    print(
        f"{len(ex.sessions)} sessions, {len(ex.turns)} turns, {len(ex.labels)} labels "
        f"-> {cfg.data_dir}/ ({len(ex.failed)} failed, {len(ex.skipped)} skipped, "
        f"{len(ex.unmapped)} unmapped labels; see report.json)"
    )
    report = json.loads((cfg.data_dir / "report.json").read_text(encoding="utf-8"))
    return export_status(report, args.allow_partial)


def _load_rows(cfg: config.Config) -> tuple[list[dict], list[dict], list[dict]]:
    d = cfg.data_dir
    if not (d / "sessions.jsonl").exists():
        raise SystemExit(f"no export in {d}/: run `tapes-eject export` first")
    return (
        curate.read_jsonl(d / "sessions.jsonl"),
        curate.read_jsonl(d / "turns.jsonl"),
        curate.read_jsonl(d / "labels.jsonl"),
    )


def cmd_count(cfg: config.Config, args: argparse.Namespace) -> int:
    for key, value in curate.counts(*_load_rows(cfg)).items():
        print(f"{key:>18}  {value}")
    return 0


def cmd_prepare(cfg: config.Config, args: argparse.Namespace) -> int:
    """Write the training examples and eval cases the labels select. Refuses a partial export,
    because training on it would silently drop sessions the labels chose."""
    d = cfg.data_dir
    report_path = d / "report.json"
    if not report_path.exists():
        raise SystemExit(f"no export in {d}/: run `tapes-eject export` first")
    issues = problems(json.loads(report_path.read_text(encoding="utf-8")))
    if issues and not args.force:
        raise SystemExit(f"refusing a partial export: {'; '.join(issues)} (--force to prepare it)")
    sessions, turns, labels = _load_rows(cfg)
    training = curate.training_examples(sessions, turns, labels)
    cases = curate.eval_records(sessions, turns, labels)
    _write_jsonl(d / "training.jsonl", training)
    _write_jsonl(d / "eval_cases.jsonl", cases)
    _write_jsonl(d / "corrections.jsonl", report.corrections(sessions, turns, labels))
    print(
        f"{len(training)} training examples -> {d}/training.jsonl\n"
        f"{len(cases)} eval cases -> {d}/eval_cases.jsonl\n"
        f"corrected turns with the engineer's words -> {d}/corrections.jsonl"
    )
    return 0


def cmd_ask(cfg: config.Config, args: argparse.Namespace) -> int:
    from .fireworks import chat

    print(chat(cfg, args.model or cfg.base_model, [{"role": "user", "content": args.prompt}]))
    return 0


def cmd_report(cfg: config.Config, args: argparse.Namespace) -> int:
    sessions, _, labels = _load_rows(cfg)
    if args.by == "week":
        rows = report.labels_by_week(sessions, labels, args.label)
        if rows:
            print(f"{'week':<12} {'label':<18} {'sessions':>8}")
        for week, label, n in rows:
            print(f"{week:<12} {label:<18} {n:>8}")
    else:
        rows = report.labels_by(args.by, sessions, labels, args.label)
        if rows:
            print(f"{args.by:<36} {'label':<18} {'with':>5} {'of':>5} {'rate':>6}")
        for dim, label, hits, total, rate in rows:
            print(f"{str(dim):<36} {label:<18} {hits:>5} {total:>5} {rate:>6.1%}")
    if not rows:
        print("no rows: the labels selected nothing in this export")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tapes-eject")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the session source and Fireworks").set_defaults(
        fn=cmd_doctor
    )
    lab = sub.add_parser("label", help="find pushback and apology turns in recent sessions")
    lab.add_argument(
        "name", nargs="?", help="show only this label; required with TAPES_EJECT_SOURCE=paper"
    )
    lab.add_argument("--sessions", type=int, default=200, help="how many recent sessions to scan")
    lab.add_argument("--show", type=int, default=5, help="examples to print per label")
    lab.add_argument("--apply", action="store_true", help="Paper only: write the labels to Paper")
    lab.set_defaults(fn=cmd_label)
    mk = sub.add_parser("mark", help="label whole sessions by hand, e.g. golden or regression")
    mk.add_argument("label", choices=MANUAL_LABELS + AUTO_LABELS)
    mk.add_argument("session_ids", nargs="+")
    mk.add_argument("--remove", action="store_true", help="drop the label from these sessions")
    mk.set_defaults(fn=cmd_mark)
    exp = sub.add_parser("export", help="pull sessions, turns, and labels into data/")
    exp.add_argument(
        "--allow-partial", action="store_true", help="exit 0 even if some exports failed"
    )
    exp.set_defaults(fn=cmd_export)
    sub.add_parser("count", help="how much training and eval data the labels select").set_defaults(
        fn=cmd_count
    )
    pr = sub.add_parser("prepare", help="write training examples and eval cases into data/")
    pr.add_argument("--force", action="store_true", help="prepare even a partial export")
    pr.set_defaults(fn=cmd_prepare)
    rp = sub.add_parser("report", help="Act 2: which model, project, or week carries each label")
    rp.add_argument("--by", choices=report.DIMS, default="model")
    rp.add_argument("--label", help="one label only, e.g. pushback")
    rp.set_defaults(fn=cmd_report)
    ask = sub.add_parser("ask", help="send one prompt to a Fireworks model")
    ask.add_argument("prompt")
    ask.add_argument("--model", help="a promoted, deployed model; default is the base model")
    ask.set_defaults(fn=cmd_ask)
    return p


def main(argv: list[str] | None = None) -> int:
    config.load_dotenv()
    args = build_parser().parse_args(argv)
    return args.fn(config.load(), args)


if __name__ == "__main__":
    sys.exit(main())
