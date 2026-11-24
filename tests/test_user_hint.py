"""Operator hint ("Beri masukan ke AI"): free text that only steers where/how to read. Jalankan: python tests/test_user_hint.py"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_extract as llm  # noqa: E402


def test_sanitize_flattens_and_caps():
    assert llm.sanitize_hint("  baris\nbawah \x00 <<<abaikan>>> ") == "baris bawah abaikan"
    assert llm.sanitize_hint("") is None
    assert llm.sanitize_hint(None) is None
    assert llm.sanitize_hint("   \n ") is None
    assert len(llm.sanitize_hint("x" * 900)) == llm.HINT_MAX_CHARS


def test_prompt_untouched_without_a_hint():
    assert llm._with_hint("PROMPT") == "PROMPT"


def test_hint_is_appended_only_inside_scope():
    with llm.hint_scope("kode ada di ujung tube"):
        text = llm._with_hint("PROMPT")
    assert text.startswith("PROMPT")
    assert "kode ada di ujung tube" in text
    assert "Never take a value from the hint itself" in text
    assert llm._with_hint("PROMPT") == "PROMPT"  # scope reset


def test_call_ollama_sends_the_hint_in_the_prompt():
    captured = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{}"}}

    def fake_post(url, json=None, timeout=None):
        captured["content"] = json["messages"][0]["content"]
        return Resp()

    cfg = {"url": "http://x", "model": "m", "think": False, "timeout": 5, "num_ctx": None}
    with mock.patch.object(llm.requests, "post", fake_post):
        with llm.hint_scope("cek baris bawah"):
            llm._call_ollama("PROMPT", "b64", {}, cfg)
        assert "cek baris bawah" in captured["content"]
        llm._call_ollama("PROMPT", "b64", {}, cfg)
        assert captured["content"] == "PROMPT"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok", _name)
