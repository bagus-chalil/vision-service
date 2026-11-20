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
assert {"ipc_client_issues", "trial_summary"} <= {k["kind"] for k in r.json()["kinds"]}, r.text

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

# ipc_client_issues: not about a decision, so "reject"/"hold" are ordinary words and pass;
# its prompt is built here (the consumer sends facts only) and carries no approval advisory rule.
ISSUE_FACTS = ["Kendala terbuka 7 hari terakhir.", '1. [js] "Camera error" | halaman: /batches/{n}/packing-check | 5 kejadian, 2 user']
llm_extract.chat_text = fake
r = client.post(URL, json={"kind": "ipc_client_issues", "facts": ISSUE_FACTS})
assert r.status_code == 200 and r.json()["kind"] == "ipc_client_issues", r.text
assert "log kendala" in seen["system"] and "Jangan menyarankan approve" not in seen["system"]
assert "- 1. [js]" in seen["prompt"]
for ok in ["Server sering reject upload foto; cek hold pada proxy.", "Sebaiknya cek browser lama di tablet Redmi."]:
    llm_extract.chat_text = lambda system, prompt, ok=ok: ok
    assert client.post(URL, json={"kind": "ipc_client_issues", "facts": ISSUE_FACTS}).status_code == 200, ok
# ... while the old kind still refuses the same kind of sentence
llm_extract.chat_text = lambda system, prompt: "Sebaiknya di-reject saja."
assert client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS}).status_code == 422


# trial_summary (approver / reviewer of the Trial app) and the optional `language`
TRIAL_FACTS = ["Trial TRIAL-1 untuk produk Tube.", "1 parameter NOT OK: Appearance (hasil: bintik; catatan: ada white spot)."]
llm_extract.chat_text = fake
r = client.post(URL, json={"kind": "trial_summary", "facts": TRIAL_FACTS})
assert r.status_code == 200 and r.json()["kind"] == "trial_summary", r.text
assert "Jangan menyarankan approve" in seen["system"]  # about a decision, so the rule is in its prompt
assert "Tulis jawaban dalam" not in seen["system"]  # language omitted = default id, prompt unchanged
assert "- Trial TRIAL-1" in seen["prompt"]

for code, name in [("en", "English"), ("ko", "Korean")]:
    r = client.post(URL, json={"kind": "trial_summary", "facts": TRIAL_FACTS, "language": code})
    assert r.status_code == 200, r.text
    assert f"Tulis jawaban dalam {name}" in seen["system"], seen["system"]
r = client.post(URL, json={"kind": "trial_summary", "facts": TRIAL_FACTS, "language": "id"})
assert r.status_code == 200 and "Tulis jawaban dalam" not in seen["system"]
assert client.post(URL, json={"kind": "trial_summary", "facts": TRIAL_FACTS, "language": "xx"}).status_code == 400

# the advisory guard also covers an English / Korean answer for the decision kinds ...
for bad, lang in [
    ("The trial looks fine, so it should be approved.", "en"),
    ("I recommend to reject this trial.", "en"),
    ("승인을 권장합니다.", "ko"),
    ("보류하는 것이 좋습니다.", "ko"),
]:
    llm_extract.chat_text = lambda system, prompt, bad=bad: bad
    r = client.post(URL, json={"kind": "trial_summary", "facts": TRIAL_FACTS, "language": lang})
    assert r.status_code == 422, (bad, r.text)
# ... but a plain statement of fact is fine
for ok, lang in [
    ("One parameter is NOT OK: Appearance. Reviewers noted white spots.", "en"),
    ("Appearance 항목이 NOT OK 입니다. 검토자는 백색 반점을 언급했습니다.", "ko"),
    ("Satu parameter NOT OK: Appearance. Review PRD mencatat bintik putih.", "id"),
]:
    llm_extract.chat_text = lambda system, prompt, ok=ok: ok
    assert client.post(URL, json={"kind": "trial_summary", "facts": TRIAL_FACTS, "language": lang}).status_code == 200, ok
# an old consumer that sends no `language` is untouched, and the Indonesian guard still works
llm_extract.chat_text = lambda system, prompt: "Disarankan reject karena ada NOT OK."
assert client.post(URL, json={"kind": "trial_summary", "facts": TRIAL_FACTS}).status_code == 422


# Ollama down / empty answer -> 503
def down(system, prompt):
    raise llm_extract.LlmUnavailable("ConnectionError: down")


llm_extract.chat_text = down
r = client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS})
assert r.status_code == 503 and r.json()["status"] == "ERROR", r.text
llm_extract.chat_text = lambda system, prompt: "   "
assert client.post(URL, json={"kind": "ipc_batch_summary", "facts": FACTS}).status_code == 503

print("insights OK")
