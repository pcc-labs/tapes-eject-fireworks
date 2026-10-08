import json

from tapes_eject.config import load
from tapes_eject.export import (
    choose_sessions,
    label_rows,
    problems,
    run_export,
    scrub,
    turn_rows,
    write_export,
)
from tapes_eject.paper import PaperError
from tapes_eject.session import parse_session
from tests.helpers import record

CFG = load(
    {
        "TAPES_EJECT_SAMPLE_SESSIONS": "5",
        "TAPES_EJECT_EXPORT_PAUSE": "0",
    }
)
PAUSED = load(
    {
        **{k: v for k, v in CFG.__dict__.items() if False},
        "TAPES_EJECT_SAMPLE_SESSIONS": "5",
    }
)


class FakePaper:
    def __init__(self, labels, attachments, by_label, recent, records, fail=()):
        self._labels, self._att, self._by_label = labels, attachments, by_label
        self._recent, self._records, self._fail = recent, records, set(fail)
        self.exported = []

    def labels(self):
        return self._labels

    def attachments(self, label_id, primitive_type):
        return self._att.get((label_id, primitive_type), [])

    def sessions(self, label=None, since=None, limit=200):
        return self._by_label.get(label, []) if label else self._recent

    def export_session(self, session_id, timeout=300):
        self.exported.append(session_id)
        if session_id in self._fail:
            raise PaperError("paperctl sessions export timed out after 300s")
        return self._records.get(session_id)


def item(sid, turns=3, seen="2026-09-01T00:00:00Z", output_tokens=0):
    rollup = {"turn_count": turns, "usage": {"output_tokens": output_tokens}}
    return {"id": sid, "last_seen_at": seen, "rollup": rollup}


def att(ptype, pid):
    return {"primitive_type": ptype, "primitive_id": pid}


def test_label_rows_map_trace_and_span_to_their_session():
    rows, unmapped = label_rows(
        {"pushback": [att("trace", "trc_1"), att("span", "spn_trc_1"), att("session", "s1")]},
        trace_session={"trc_1": "s1"},
        span_trace={"spn_trc_1": "trc_1"},
    )
    assert [(r["primitive_type"], r["session_id"], r["turn_id"]) for r in rows] == [
        ("trace", "s1", "trc_1"),
        ("span", "s1", "trc_1"),
        ("session", "s1", None),
    ]
    assert unmapped == []


def test_unmapped_attachment_is_kept_with_null_session_and_reported():
    rows, unmapped = label_rows({"pushback": [att("trace", "trc_gone")]}, {}, {})
    assert rows == unmapped
    assert rows[0]["session_id"] is None and rows[0]["turn_id"] == "trc_gone"


def test_label_rows_skip_non_session_primitives():
    rows, _ = label_rows({"paper": [att("skill", "sk_1")]}, {}, {})
    assert rows == []


def test_choose_sessions_skips_oversized_labeled_and_samples_recent_under_the_caps():
    labeled = {"big": item("big", 300), "s1": item("s1", 4)}
    recent = [item("s1"), item("r0", 0), item("r1", 1), item("r2", 10), item("r3", 160)]
    ids, skipped = choose_sessions(labeled, recent, sample=2, max_turns=150)
    assert ids == ["s1", "r1", "r2"]  # empty and over-cap sessions are never sampled
    assert skipped == [("big", "300 turns > max 150")]


def test_choose_sessions_skips_short_sessions_with_huge_output():
    """Three turns but 800k output tokens: the shape that can overload Paper's export."""
    labeled = {"whale": item("whale", 3, output_tokens=800_000), "s1": item("s1", 4)}
    recent = [item("r1", 10, output_tokens=2_744_041), item("r2", 10, output_tokens=295_000)]
    ids, skipped = choose_sessions(labeled, recent, sample=5, max_turns=150)
    assert ids == ["s1", "r2"]
    assert skipped == [("whale", "800000 output tokens > max 400000")]
    assert choose_sessions(labeled, recent, 5, 150, max_output_tokens=900_000)[0] == [
        "whale",
        "s1",
        "r2",
    ]


def test_choose_sessions_honours_the_skip_list():
    labeled = {"poison": item("poison", 9, output_tokens=201_633), "s1": item("s1", 4)}
    ids, skipped = choose_sessions(labeled, [item("poison")], 5, 150, skip=frozenset({"poison"}))
    assert ids == ["s1"]
    assert skipped == [("poison", "in TAPES_EJECT_SKIP_SESSIONS: its export overloads Paper")]


def test_session_row_carries_the_project_from_the_cwd():
    from tapes_eject.export import project_name, session_row

    assert (
        project_name("/home/someone/code/pcc-labs/tapes-eject-fireworks") == "tapes-eject-fireworks"
    )
    assert (
        project_name("/repo/") == "repo" and project_name(None) is None and project_name("") is None
    )
    rec = record("s1", [("trc_1", "hi", "ok")])
    rec["session"]["cwd"] = "/Users/x/Projects/site"
    assert session_row(parse_session(rec), set())["project"] == "site"


def test_turn_rows_are_redacted():
    sess = parse_session(
        record("s1", [("trc_1", "use key sk-ant-abcdefghijklmnopqrstuvwxyz0123", "ok")])
    )
    (row,) = turn_rows(sess)
    assert "sk-ant-" not in row["user_prompt"] and "[redacted]" in row["user_prompt"]
    assert row["agent_text"] == "ok"


def _paper(fail=()):
    return FakePaper(
        labels=[{"id": "L1", "name": "pushback", "usage": {"session": 1, "trace": 2}}],
        attachments={
            ("L1", "session"): [att("session", "s1")],
            ("L1", "trace"): [att("trace", "trc_b"), att("trace", "trc_elsewhere")],
        },
        by_label={"pushback": [item("s1")]},
        recent=[item("s1"), item("s2"), item("s3")],
        records={
            "s1": record("s1", [("trc_a", "do x", "did x"), ("trc_b", "no, do y", "did y")]),
            "s2": record("s2", [("trc_c", "hello", "hi"), ("trc_d", "thanks", "np")]),
        },
        fail=fail,
    )


def test_run_export_reads_only_the_cache_after_an_outage(tmp_path):
    # A warm pass caches s1 and s2. Then s1 is dropped from the cache and its fetch fails
    # outage-shaped: s2 still comes from the cache, s3 is not tried.
    warm = _paper()
    run_export(warm, CFG, log=lambda m: None, cache_dir=tmp_path)
    (tmp_path / "s1.json").unlink()
    paper = _paper(fail={"s1"})
    ex = run_export(paper, CFG, log=lambda m: None, cache_dir=tmp_path)
    assert paper.exported == ["s1"]  # nothing was requested after the outage
    assert {s["session_id"] for s in ex.sessions} == {"s2"}
    assert ex.failed == [
        ("s1", "paperctl sessions export timed out after 300s"),
        ("s3", "not tried: export service unreachable"),
    ]
    assert "1 session exports failed" not in problems(
        {"failed": ex.failed, "sessions": 0, "outcome_unknown": 0}
    )


def test_run_export_pauses_after_each_fetch_but_not_after_a_cache_hit(tmp_path):
    paper = _paper()
    naps: list[float] = []
    run_export(paper, PAUSED, log=lambda m: None, cache_dir=tmp_path, sleep=naps.append)
    assert paper.exported == ["s1", "s2", "s3"] and naps == [1.0, 1.0, 1.0]
    paper2 = _paper()
    naps2: list[float] = []
    run_export(paper2, PAUSED, log=lambda m: None, cache_dir=tmp_path, sleep=naps2.append)
    assert paper2.exported == ["s3"] and naps2 == [1.0]  # s3 has no record, so it is never cached


def test_run_export_continues_past_a_failed_export_and_reports_it(tmp_path):
    ex = run_export(_paper(fail={"s3"}), CFG, log=lambda m: None)
    assert {s["session_id"] for s in ex.sessions} == {"s1", "s2"}
    assert ex.failed == [("s3", "paperctl sessions export timed out after 300s")]
    assert {s["session_id"]: s["has_outcome"] for s in ex.sessions} == {"s1": True, "s2": True}
    assert [u["primitive_id"] for u in ex.unmapped] == ["trc_elsewhere"]
    write_export(ex, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failed"] == [["s3", "paperctl sessions export timed out after 300s"]]
    assert report["unmapped"] == 1 and report["outcome_unknown"] == 0
    assert len((tmp_path / "labels.jsonl").read_text().splitlines()) == 3


def test_span_ids_in_papers_trace_tilde_span_shape_map_through_their_trace():
    rows, unmapped = label_rows(
        {"missing-knowledge": [att("span", "trc_1~llm_9")]}, {"trc_1": "s1"}, {}
    )
    assert unmapped == []
    assert (rows[0]["session_id"], rows[0]["turn_id"], rows[0]["span_id"]) == (
        "s1",
        "trc_1",
        "trc_1~llm_9",
    )


def test_oversized_sessions_are_skipped_not_failed(tmp_path):
    paper = _paper()
    paper._by_label["pushback"] = [item("s1", turns=500)]
    ex = run_export(paper, CFG, log=lambda m: None)
    assert ex.skipped == [("s1", "500 turns > max 150")]
    assert ex.failed == []


def test_problems_flag_failed_exports_empty_exports_and_unknown_outcomes():
    ok = {"sessions": 3, "failed": [], "outcome_unknown": 0}
    assert problems(ok) == []
    assert problems({**ok, "failed": [["s3", "timed out"]]}) == ["1 session exports failed"]
    assert problems({**ok, "sessions": 0}) == ["no sessions exported"]
    assert problems({**ok, "outcome_unknown": 2}) == ["outcome unknown for 2 sessions"]


def test_papers_own_no_outcome_label_counts_as_no_outcome():
    paper = _paper()
    paper._labels.append({"id": "L2", "name": "no-outcome", "usage": {"session": 1}})
    paper._att[("L2", "session")] = [att("session", "s1")]
    ex = run_export(paper, CFG, log=lambda m: None)
    assert {s["session_id"]: s["has_outcome"] for s in ex.sessions}["s1"] is False


def test_an_unchanged_session_is_read_from_the_cache_on_the_next_export(tmp_path):
    first = _paper()
    run_export(first, CFG, log=lambda m: None, cache_dir=tmp_path)
    assert sorted(first.exported) == ["s1", "s2", "s3"]
    again = _paper()
    ex = run_export(again, CFG, log=lambda m: None, cache_dir=tmp_path)
    assert again.exported == ["s3"]  # s3 has no record, so nothing was cached for it
    assert {s["session_id"] for s in ex.sessions} == {"s1", "s2"}


def test_a_changed_session_is_exported_again(tmp_path):
    run_export(_paper(), CFG, log=lambda m: None, cache_dir=tmp_path)
    moved = _paper()
    moved._recent[1] = item("s2", seen="2026-09-02T00:00:00Z")
    run_export(moved, CFG, log=lambda m: None, cache_dir=tmp_path)
    assert moved.exported == ["s2", "s3"]


def test_scrub_catches_secrets_the_parser_misses():
    text = (
        '{"api_key": "abcd1234efgh5678ijkl"} sk_live_abcdefghijklmnop '
        "postgres://admin:S3cr3tPassw0rd@db.example:5432/app "
        "dapi0123456789abcdef0123456789abcdef"
    )
    out = scrub(text)
    for secret in ("abcd1234efgh5678ijkl", "sk_live_abcdef", "S3cr3tPassw0rd", "dapi0123"):
        assert secret not in out
    assert "db.example" in out


def test_turn_rows_apply_the_extra_scrub():
    sess = parse_session(
        record("s1", [("trc_1", "connect to postgres://u:hunter2hunter2@h/db", "ok")])
    )
    (row,) = turn_rows(sess)
    assert "hunter2hunter2" not in row["user_prompt"]


def test_parser_drops_scaffolding_flags_synthetic_turns_and_redacts():
    from tapes_eject.session import parse_session

    rec = record(
        "s1",
        [
            ("t1", "<system-reminder>x</system-reminder>fix it, token=abcdefghijklmnop", "done"),
            ("t2", "[Request interrupted by user]", ""),
        ],
    )
    sess = parse_session(rec)
    assert sess.turns[0].user_prompt == "fix it, token=[redacted]"
    assert sess.turns[0].assistant_text == ["done"] and not sess.turns[0].synthetic
    assert sess.turns[1].synthetic
    assert parse_session({"session": {"id": "e"}, "traces": []}) is None
