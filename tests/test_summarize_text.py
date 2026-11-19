"""Quick check of POST /api/summarize-text with Ollama mocked (run: LOAD_PADDLEOCR=0 python tests/test_summarize_text.py)."""
import os
import sys

os.environ["LOAD_PADDLEOCR"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import llm_extract  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)

llm_extract.chat_text = lambda system, prompt: f"ringkasan: {prompt[:5]}"
r = client.post("/api/summarize-text", json={"system": "s", "prompt": "halo dunia"})
assert r.status_code == 200 and r.json() == {"status": "OK", "text": "ringkasan: halo "}, r.text

assert client.post("/api/summarize-text", json={"prompt": "   "}).status_code == 400
assert client.post("/api/summarize-text", json={"prompt": "x" * 12001}).status_code == 400


def down(system, prompt):
    raise llm_extract.LlmUnavailable("ConnectionError: down")


llm_extract.chat_text = down
r = client.post("/api/summarize-text", json={"prompt": "halo"})
assert r.status_code == 503 and r.json()["status"] == "ERROR", r.text
print("summarize-text OK")
