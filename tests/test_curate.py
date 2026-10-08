from tapes_eject.curate import (
    correction_cases,
    counts,
    eval_records,
    is_holdout,
    session_cases,
    training_examples,
)


def s(sid, has_outcome=True, title="t"):
    return {"session_id": sid, "has_outcome": has_outcome, "title": title}


def t(sid, tid, ordinal, prompt, reply, synthetic=False):
    return {
        "session_id": sid,
        "turn_id": tid,
        "ordinal": ordinal,
        "user_prompt": prompt,
        "agent_text": reply,
        "synthetic": synthetic,
    }


def lab(label, sid, ptype="session", tid=None):
    return {"label": label, "primitive_type": ptype, "session_id": sid, "turn_id": tid}


TURNS = [
    t("a", "a1", 0, "add a flag", "added --x"),
    t("a", "a2", 1, "no, call it --y", "renamed to --y"),
    t("b", "b1", 0, "write tests", "wrote tests"),
    t("c", "c1", 0, "ship it", "shipped"),
]

# golden sessions split by a stable hash: g1 is held out for eval, g0 is training data
HELD, TRAINED = "g1", "g0"
GOLDEN_TURNS = [
    t(HELD, "h1", 0, "migrate the table", "wrote a reversible migration"),
    t(TRAINED, "t1", 0, "bump the version", "bumped to 0.2.0"),
]


def test_training_takes_outcome_sessions_without_negative_labels():
    ex = training_examples(
        [s("a"), s("b"), s("c", has_outcome=False)], TURNS, [lab("pushback", "a")]
    )
    assert [e["session_id"] for e in ex] == ["b"]
    assert ex[0]["messages"] == [
        {"role": "user", "content": "write tests"},
        {"role": "assistant", "content": "wrote tests"},
    ]


def test_golden_is_included_without_outcome_but_regression_always_excludes():
    labels = [lab("golden", "c"), lab("golden", "b"), lab("regression", "b")]
    ex = training_examples([s("b"), s("c", has_outcome=False)], TURNS, labels)
    assert [e["session_id"] for e in ex] == ["c"]


def test_unknown_outcome_is_not_training_eligible():
    assert training_examples([s("b", has_outcome=None)], TURNS, []) == []


def test_correction_case_is_the_context_before_the_corrected_turn():
    (case,) = correction_cases(TURNS, [lab("pushback", "a", "trace", "a2")])
    assert case["inputs"]["messages"] == [{"role": "user", "content": "add a flag"}]
    (g,) = case["expectations"]["guidelines"]
    assert "no, call it --y" in g


def test_first_turn_corrections_are_skipped_and_duplicates_collapse():
    labels = [
        lab("pushback", "b", "trace", "b1"),
        lab("pushback", "a", "trace", "a2"),
        lab("pushback", "a", "span", "a2"),
        lab("observation", "a", "trace", "a2"),
        lab("pushback", None, "trace", "gone"),
    ]
    assert len(correction_cases(TURNS, labels)) == 1


def test_regression_session_case_quotes_its_corrections():
    labels = [lab("regression", "a"), lab("pushback", "a", "trace", "a2")]
    (case,) = session_cases([s("a")], TURNS, labels)
    assert case["inputs"]["messages"] == [{"role": "user", "content": "add a flag"}]
    assert "no, call it --y" in case["expectations"]["guidelines"][0]


def test_golden_session_case_quotes_the_good_answer():
    (case,) = session_cases([s(HELD)], GOLDEN_TURNS, [lab("golden", HELD)])
    assert "wrote a reversible migration" in case["expectations"]["guidelines"][0]


def test_counts():
    c = counts(
        [s("a"), s("b"), s("c", has_outcome=None)],
        TURNS,
        [lab("pushback", "a", "trace", "a2"), lab("golden", "b")],
    )
    assert c == {
        "sessions": 3,
        "with_outcome": 2,
        "outcome_unknown": 1,
        "training_examples": 1,
        "correction_cases": 1,
        "golden": 1,
        "regression": 0,
        "eval_cases": 1,
    }


def test_holdout_is_stable_and_takes_about_one_in_five():
    assert is_holdout(HELD) and not is_holdout(TRAINED)
    share = sum(is_holdout(f"s{i}") for i in range(1000)) / 1000
    assert 0.15 < share < 0.25


def test_a_golden_session_is_training_data_or_an_eval_case_never_both():
    sessions = [s(HELD), s(TRAINED)]
    labels = [lab("golden", HELD), lab("golden", TRAINED)]
    trained = {e["session_id"] for e in training_examples(sessions, GOLDEN_TURNS, labels)}
    cases = session_cases(sessions, GOLDEN_TURNS, labels)
    assert trained == {TRAINED}
    assert len(cases) == 1 and "reversible migration" in cases[0]["expectations"]["guidelines"][0]


def test_sessions_with_any_correction_label_are_not_training_data():
    labels = [lab("observation", "b", "trace", "b1")]
    assert training_examples([s("b")], TURNS, labels) == []


def test_eval_records_merge_cases_that_share_inputs():
    labels = [lab("regression", "a"), lab("pushback", "a", "trace", "a2")]
    (case,) = eval_records([s("a")], TURNS, labels)
    assert case["inputs"]["messages"] == [{"role": "user", "content": "add a flag"}]
    guidelines = case["expectations"]["guidelines"]
    assert len(guidelines) == 2
    assert any("without being told" in g for g in guidelines)
    assert any("must not repeat" in g for g in guidelines)
