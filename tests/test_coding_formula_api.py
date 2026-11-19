"""POST /api/analyze field_type=coding_formula dengan Ollama tiruan.
Jalankan: python tests/test_coding_formula_api.py (LOAD_PADDLEOCR=0 otomatis)."""
import os
import sys

os.environ["LOAD_PADDLEOCR"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import llm_extract  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)
ok_png = cv2.imencode(".png", np.full((40, 60, 3), 255, np.uint8))[1].tobytes()

# Foto 1 (belakang): EXP jelas, batch berubah antar view; foto 2 (depan): batch jelas.
script = {}


import json  # noqa: E402


def fake_call(prompt, image_b64, schema, cfg):
    assert cfg.get("num_predict"), "jalur coding_formula wajib membatasi token keluaran"
    item = script["queue"].pop(0)
    content = item if isinstance(item, str) else json.dumps({"lines": item})
    return {"message": {"content": content}, "model": "fake", "done_reason": "length"}


llm_extract._call_ollama = fake_call
os.environ["CODE_LINES_VIEW_SIDES"] = "1024,640,1600,2048"  # tes memakai 4 view

back = [["BATCH NO PL8", "EXP DATE 181129 TQ"], ["FLR", "181129 TQ"], ["PLB", "181129 TQ"], ["PLB", "181129 TQ"]]
front = [["PLB"]] * 4
script["queue"] = [list(v) for v in back] + [list(v) for v in front]
files = {"file": ("b.png", ok_png, "image/png"), "file2": ("f.png", ok_png, "image/png")}
r = client.post("/api/analyze", files=files,
                data={"field_type": "coding_formula", "coding_formula": "label_exp_ddmmyy_init"})
j = r.json()
assert r.status_code == 200 and j["status"] == "OK", j
assert j["fields"]["exp"]["value"] == "181129" and j["fields"]["exp"]["date"]["day"] == 18, j
assert j["fields"]["batch"]["value"] == "PLB" and j["fields"]["batch"]["photos"] == [1], j
assert j["photos"][0]["fields"]["batch"]["status"] == "LOW_CONFIDENCE", j
assert "key" not in j["fields"]["exp"], "key internal tidak boleh bocor"

# Konflik antar foto -> CONFLICT, bukan OK
script["queue"] = [["OKB EXP 231029"]] * 4 + [["AKB EXP 231029"]] * 4
r = client.post("/api/analyze", files=files,
                data={"field_type": "coding_formula", "coding_formula": '["{BATCH3} EXP {DD}{MM}{YY}"]'})
j = r.json()
assert j["status"] == "LOW_CONFIDENCE" and j["fields"]["batch"]["status"] == "CONFLICT", j

# Early exit: view ke-2 sudah beda di semua field wajib -> berhenti (2 panggilan, bukan 4)
script["queue"] = [["OKB EXP 231029"], ["AKB EXP 231028"], ["x"], ["x"]]
r = client.post("/api/analyze", files={"file": ("a.png", ok_png, "image/png")},
                data={"field_type": "coding_formula", "coding_formula": "{BATCH3} EXP {DD}{MM}{YY}"})
j = r.json()
assert j["attempts"] == 2 and len(script["queue"]) == 2 and j["status"] == "LOW_CONFIDENCE", j

# View yang loop/terpotong (JSON rusak) = view gagal: field tidak OK, tapi bukan ERROR
script["queue"] = [["OKB EXP 231029"], '{"lines": ["OKB", "OKB", "OKB"', ["OKB EXP 231029"], ["OKB EXP 231029"]]
r = client.post("/api/analyze", files={"file": ("a.png", ok_png, "image/png")},
                data={"field_type": "coding_formula", "coding_formula": "{BATCH3} EXP {DD}{MM}{YY}"})
j = r.json()
assert j["status"] == "LOW_CONFIDENCE" and "error" in j["photos"][0]["views_read"][1], j
assert j["attempts"] == 2, "tanpa retry & berhenti begitu semua field wajib mustahil OK"

# Formula salah / kosong -> 400
for bad in ("{FOO}", "[not json"):
    r = client.post("/api/analyze", files={"file": ("a.png", ok_png, "image/png")},
                    data={"field_type": "coding_formula", "coding_formula": bad})
    assert r.status_code == 400 and r.json()["status"] == "ERROR", (bad, r.text)

# Ollama mati -> ERROR, bukan OK
def down(*a, **k):
    raise llm_extract.LlmUnavailable("ConnectionError: down")


llm_extract._call_ollama = down
r = client.post("/api/analyze", files={"file": ("a.png", ok_png, "image/png")},
                data={"field_type": "coding_formula", "coding_formula": "label_exp_ddmmyy_init"})
assert r.json()["status"] == "ERROR", r.text

# Tanpa formula = AUTO (gabungan katalog), ditandai formula_source
llm_extract._call_ollama = fake_call
script["queue"] = [["OKB EXP 112028"]] * 4
r = client.post("/api/analyze", files={"file": ("a.png", ok_png, "image/png")},
                data={"field_type": "coding_formula"})
j = r.json()
assert j["formula_source"] == "auto" and j["status"] == "OK", j
assert j["fields"]["exp"]["date"] == {"year": 2028, "month": 11, "day": None}, j  # MMYYYY, bukan DDMMYY
script["queue"] = [["EXP 08/2026", "EXP 04/2029"]] * 4  # dua tanggal ber-label EXP beda -> AMBIGUOUS
r = client.post("/api/analyze", files={"file": ("a.png", ok_png, "image/png")},
                data={"field_type": "coding_formula", "coding_formula": "auto"})
assert r.json()["fields"]["exp"]["status"] == "LOW_CONFIDENCE", r.json()

cat = client.get("/api/coding-formulas").json()
assert "BATCH3" in cat["tokens"] and "tube_exp_month_name" in cat["catalog"], cat
assert any(f["key"] == "coding_formula" and f.get("needs_formula") for f in client.get("/api/field-types").json())
print("coding_formula API OK")
