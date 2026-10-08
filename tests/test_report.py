import pytest

from tapes_eject.report import corrections, labels_by, labels_by_week, week_of

SESSIONS = [
    {"session_id": "a", "model": "opus", "project": "p1", "started_at": "2026-09-30T10:00:00Z"},
    {"session_id": "b", "model": "opus", "project": "p2", "started_at": "2026-10-02T10:00:00Z"},
    {"session_id": "c", "model": "sonnet", "project": "p1", "started_at": "2026-10-05T09:00:00Z"},
    {"session_id": "d", "model": "sonnet", "project": "p1", "started_at": None},
]
LABELS = [
    {"label": "pushback", "session_id": "a", "turn_id": "a2"},
    {"label": "pushback", "session_id": "a", "turn_id": "a3"},  # two turns, one session
    {"label": "pushback", "session_id": "c", "turn_id": "c2"},
    {"label": "golden", "session_id": "b", "turn_id": None},
]


def test_labels_by_model_counts_sessions_over_every_exported_session():
    rows = labels_by("model", SESSIONS, LABELS)
    assert rows == [
        ("opus", "golden", 1, 2, 0.5),
        ("opus", "pushback", 1, 2, 0.5),
        ("sonnet", "pushback", 1, 2, 0.5),
    ]
    assert labels_by("project", SESSIONS, LABELS, only="pushback") == [
        ("p1", "pushback", 2, 3, 0.667)
    ]
    with pytest.raises(ValueError):
        labels_by("user", SESSIONS, LABELS)


def test_labels_by_week_starts_on_monday_and_skips_undated_sessions():
    assert week_of("2026-10-04T23:00:00Z") == "2026-09-28"
    assert labels_by_week(SESSIONS, LABELS) == [
        ("2026-09-28", "golden", 1),
        ("2026-09-28", "pushback", 1),
        ("2026-10-05", "pushback", 1),
    ]


def test_corrections_carry_the_engineers_words_and_skip_unexported_turns():
    turns = [
        {"session_id": "a", "turn_id": "a2", "ordinal": 1, "user_prompt": "no, use uv"},
        {
            "session_id": "c",
            "turn_id": "c2",
            "ordinal": 1,
            "user_prompt": "wrong file",
            "synthetic": True,
        },
    ]
    got = corrections(SESSIONS, turns, LABELS)
    assert [(r["session_id"], r["correction"], r["model"]) for r in got] == [
        ("a", "no, use uv", "opus")
    ]
