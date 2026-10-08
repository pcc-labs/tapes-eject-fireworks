import pytest

from tapes_eject.config import load, ping_url


def test_load_needs_no_settings_and_defaults_to_the_cookbook_model():
    cfg = load({})
    assert cfg.fireworks_api_key is None
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    assert cfg.judge_model == cfg.base_model
    assert cfg.fireworks_api == "https://api.fireworks.ai"
    assert cfg.sample_sessions == 200 and cfg.max_turns == 150
    assert cfg.max_output_tokens == 400_000
    assert cfg.export_pause == 1.0 and cfg.skip_sessions == frozenset()
    assert cfg.org_slug is None


def test_load_reads_overrides():
    cfg = load(
        {
            "FIREWORKS_API_KEY": "fw_x",
            "TAPES_EJECT_BASE_MODEL": "accounts/fireworks/models/m",
            "FIREWORKS_BASE_URL": "https://dev.example/",
            "AUTOLABEL_URL": "http://h:9996/v1/cassettes/autolabel/",
            "TAPES_EJECT_MAX_OUTPUT_TOKENS": "50000",
            "TAPES_EJECT_SKIP_SESSIONS": "a, b,",
        }
    )
    assert cfg.require_key() == "fw_x"
    assert cfg.base_model == "accounts/fireworks/models/m"
    assert cfg.fireworks_api == "https://dev.example"
    assert cfg.autolabel_url == "http://h:9996/v1/cassettes/autolabel"
    assert cfg.max_output_tokens == 50_000
    assert cfg.skip_sessions == {"a", "b"}


def test_require_key_names_the_missing_setting():
    with pytest.raises(SystemExit, match="FIREWORKS_API_KEY"):
        load({}).require_key()


def test_ping_url_is_the_host_root():
    assert ping_url("http://127.0.0.1:9996/v1/cassettes/autolabel") == "http://127.0.0.1:9996/ping"
    assert ping_url("https://box.example/api/autolabel") == "https://box.example/ping"
