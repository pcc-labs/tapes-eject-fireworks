from tapes_eject.config import load
from tapes_eject.doctor import _cassette, _fireworks, run_checks


def test_run_checks_reports_each_and_fails_on_any_error():
    def boom():
        raise RuntimeError("paperd not running")

    ok, lines = run_checks([("a", lambda: "fine"), ("b", boom)])
    assert ok is False
    assert lines == ["ok    a: fine", "FAIL  b: paperd not running"]


def test_run_checks_all_pass():
    ok, _ = run_checks([("a", lambda: "x")])
    assert ok is True


def test_a_missing_fireworks_key_is_a_fail_line_not_an_exit():
    ok, lines = run_checks([("fireworks", lambda: _fireworks(load({})))])
    assert ok is False and lines == [
        "FAIL  fireworks: FIREWORKS_API_KEY is not set; add it to .env"
    ]


def test_a_missing_cassette_is_not_a_failure():
    assert "optional" in _cassette("http://127.0.0.1:1/v1/cassettes/autolabel")
