from pathlib import Path

import pytest

from claude_voice import config

EXAMPLE = Path(__file__).parent.parent / "config.example.toml"


def test_example_config_matches_the_defaults():
    assert config.load(EXAMPLE) == config.Config()


def test_bad_values_are_rejected(tmp_path):
    for body, message in [
        ("[aec]\nmode = 'loud'", "aec.mode"),
        ("[vad]\nthreshold = 'high'", "vad.threshold must be float"),
        ("[stt]\nlanguges = ['en']", "unknown config key: stt.languges"),
        ("[audio]\ninput_device = 'mic; rm -rf /'", "not a valid PipeWire node name"),
        ("[vad]\nthreshold = 0.3\nend_threshold = 0.4", "end_threshold"),
        ("[tts]\nspeed = 5", "tts.speed"),
    ]:
        f = tmp_path / "c.toml"
        f.write_text(body)
        with pytest.raises(ValueError, match=message):
            config.load(f)
