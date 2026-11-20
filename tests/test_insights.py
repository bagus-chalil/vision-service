"""Check of POST /api/insights/summarize with Ollama mocked (run: LOAD_PADDLEOCR=0 python tests/test_insights.py)."""
import os
import sys

os.environ["LOAD_PADDLEOCR"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import insights  # noqa: E402
import llm_extract  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)
URL = "/api/insights/summarize"
FACTS = ["Batch 26X228/CKB, produk Lipstik A.", "2 item NOT OK: Packing Primary Bulk; Startup Product Standard."]

# kinds listing
r = client.get("/api/insights/kinds")
assert r.status_code == 200 and r.json()["kinds"][0]["kind"] == "ipc_batch_summary", r.text

# happy path: the consumer sends facts only; the prompt is built here and carries the rules.
seen = {}


def fake(system, prompt):
    seen["system"], seen["prompt"] = system, prompt
    return "Ada dua temuan NOT OK pada Packing dan Startup."


llm_extract.chat_text = fake
r = client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS})
assert r.status_code == 200 and r.json() == {"status": "OK", "kind": "ipc_batch_summary", "text": "Ada dua temuan NOT OK pada Packing dan Startup."}, r.text
assert "Jangan menyarankan approve" in seen["system"]
assert "- Batch 26X228/CKB" in seen["prompt"] and "- 2 item NOT OK" in seen["prompt"]

# bad requests
assert client.post(URL, json={"kind": "nope", "facts": FACTS}).status_code == 400
assert client.post(URL, json={"kind": "ipc_batch_summary", "facts": ["  ", ""]}).status_code == 400
assert client.post(URL, json={"kind": "ipc_batch_summary", "facts": ["x"] * (insights.MAX_FACTS + 1)}).status_code == 400
assert client.post(URL, json={"kind": "ipc_batch_summary", "facts": ["x" * (insights.MAX_FACT_CHARS + 1)]}).status_code == 400
assert client.post(URL, json={"kind": "ipc_batch_summary"}).status_code == 422  # facts missing (pydantic)

# advisory guard: an answer that recommends a decision is refused ...
for bad in [
    "Temuan banyak, sebaiknya di-approve saja.",
    "Disarankan reject karena ada NOT OK.",
    "Batch ini layak disetujui.",
    "Saya sarankan untuk hold dulu.",
]:
    llm_extract.chat_text = lambda system, prompt, bad=bad: bad
    r = client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS})
    assert r.status_code == 422 and r.json()["status"] == "ERROR", (bad, r.text)

# ... but restating a recorded decision is fine.
for ok in [
    "Keputusan QC: Packing: Hold, Finished Good: Rejected.",
    "Tidak ada item NOT OK pada batch ini.",
    "Item Packing Primary Bulk berulang di 3 dari 10 batch terakhir.",
]:
    llm_extract.chat_text = lambda system, prompt, ok=ok: ok
    r = client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS})
    assert r.status_code == 200, (ok, r.text)


# Ollama down / empty answer -> 503
def down(system, prompt):
    raise llm_extract.LlmUnavailable("ConnectionError: down")


llm_extract.chat_text = down
r = client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS})
assert r.status_code == 503 and r.json()["status"] == "ERROR", r.text
llm_extract.chat_text = lambda system, prompt: "   "
assert client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS}).status_code == 503

print("insights OK")
