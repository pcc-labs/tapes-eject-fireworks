import pytest

from tapes_eject import fireworks as fw


def test_plan_steps_smoke_run_or_epochs():
    assert fw.plan_steps(100, 8, 2.0, max_steps=3) == 3
    assert fw.plan_steps(100, 8, 2.0, max_steps=0) == 25
    assert fw.plan_steps(1, 8, 1.0, max_steps=0) == 1


def test_batch_wraps_around_the_dataset():
    assert fw.batch([1, 2, 3], 0, 2) == [1, 2]
    assert fw.batch([1, 2, 3], 1, 2) == [3, 1]


def test_checkpoint_names_fail_before_training_not_at_promote():
    assert fw.check_checkpoint_name("final-state") == "final-state"
    with pytest.raises(ValueError, match="1-17"):
        fw.check_checkpoint_name("a-much-too-long-name")


def test_find_promotable_matches_both_listing_shapes():
    cps = [
        {"name": "accounts/a/trainingSessions/s/checkpoints/run-1-final-ab12", "promotable": False},
        {"name": "accounts/a/trainingSessions/s/checkpoints/run-1-final-cd34", "promotable": True},
        {"name": "accounts/a/trainingSessions/s/checkpoints/final-state", "promotable": True},
    ]
    assert fw.find_promotable(cps, "final", "run-1")["name"].endswith("cd34")
    assert fw.find_promotable(cps[:1], "final", "run-1") is None


def test_model_id_is_unique_per_run_and_valid():
    assert fw.model_id("Tapes Eject Agent", "run-d4a0c70aac564b6e") == "tapes-eject-agent-d4a0c70a"
    assert len(fw.model_id("x" * 80, "run-12345678")) == 63
    assert fw.model_id("agent", None) == "agent"


def test_parse_verdict_reads_json_and_refuses_to_guess():
    assert fw.parse_verdict('ok {"pass": true, "why": "uses uv"}') == (True, "uses uv")
    assert fw.parse_verdict('{"pass": "yes"}')[0] is None
    assert fw.parse_verdict("I think it passes")[0] is None


def test_judge_messages_show_conversation_reply_and_guidelines():
    msgs = fw.judge_messages([{"role": "user", "content": "add a flag"}], "done", ["uses argparse"])
    assert msgs[0]["role"] == "system" and '"pass"' in msgs[0]["content"]
    assert "[user]\nadd a flag" in msgs[1]["content"]
    assert "Reply to grade:\ndone" in msgs[1]["content"] and "- uses argparse" in msgs[1]["content"]


def test_summarize_counts_only_cases_judged_for_both_models():
    scores = [
        {"base_pass": False, "tuned_pass": True},
        {"base_pass": True, "tuned_pass": True},
        {"base_pass": None, "tuned_pass": True},
    ]
    got = fw.summarize(scores)
    assert got["base"] == {"passed": 1, "of": 2, "rate": 0.5}
    assert got["tuned"] == {"passed": 2, "of": 2, "rate": 1.0}
    assert got["unjudged"] == {"cases": 1}
