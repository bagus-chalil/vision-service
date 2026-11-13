"""Evaluasi field_type tube_exp_date (PaddleOCR, jalur label_anchor "EXP") vs kunci DB, di set
tube yang sama dengan tools/evaluate_tube_exp_llm.py, supaya bisa dibandingkan foto per foto.

Tanpa import main.py (tidak memuat model lain). PaddleOCR enable_mkldnn=False (gotcha #2).
IZIN KHUSUS sesi 2026-11-13: dijalankan di laptop untuk ukuran ini saja.
Kunci = exp_date DB: OK beda DB = CEK_VISUAL (cetakan bisa memang beda dari DB).
Pakai: venv\\Scripts\\python.exe tools/evaluate_tube_exp_paddle.py --set old|new
Output: testdata_private/tube/paddle_<set>.json (gitignored, resumable)."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

import tube_emboss_pipeline as T  # noqa: E402

D = ROOT / "testdata_private" / "tube"
ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=("old", "new"), required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

OUT = D / f"paddle_{args.set}.json"
old_ids = {x["id"] for x in json.loads((D / "classify.json").read_text(encoding="utf-8"))}
items = [x for x in json.loads((D / "classify_v2.json").read_text(encoding="utf-8"))
         if x.get("container") == "tube" and x.get("code_on_crimp") and not x.get("error")
         and x.get("db_exp") not in (None, "", "NULL")
         and ((x["id"] in old_ids) if args.set == "old" else (x["id"] not in old_ids))]
if args.limit:
    items = items[: args.limit]
done = ({x["id"]: x for x in json.loads(OUT.read_text(encoding="utf-8")) if x.get("status") != "ERROR"}
        if OUT.exists() else {})

ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=True, enable_mkldnn=False)
yolo = T.load_yolo_model()
res = []
for i, it in enumerate(items, 1):
    if it["id"] in done:
        res.append(done[it["id"]])
        continue
    y, m, d = it["db_exp"].split("-")
    base = {"id": it["id"], "db_ddmmyy": d + m + y[2:], "black": bool(it.get("black_mark_over_code"))}
    img = cv2.imread(it["path"])
    if img is None:
        res.append({**base, "status": "ERROR", "error": "unreadable"})
        continue
    t0 = time.perf_counter()
    try:
        r = T.analyze_tube_emboss(img, field_type="tube_exp_date", request_id=f"eval-{it['id']}",
                                  ocr_engine=ocr, yolo_model=yolo, debug=False, image_ref=str(it["id"]))
    except Exception as e:  # noqa: BLE001
        res.append({**base, "status": "ERROR", "error": f"{type(e).__name__}: {e}"[:120]})
        continue
    res.append({**base, "status": r.get("status"), "error_reason": r.get("error_reason"),
                "extracted": r.get("extracted_date_code"), "format_valid": r.get("format_valid"),
                "confidence": r.get("confidence"), "match_method": r.get("match_method"),
                "sec": round(time.perf_counter() - t0, 2)})
    if i % 10 == 0:
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(i, len(items), flush=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
ok = [r for r in res if r.get("status") == "OK"]
match = [r for r in ok if r["extracted"] == r["db_ddmmyy"]]
secs = sorted(r["sec"] for r in res if "sec" in r)
print(f"[paddle {args.set}] n={len(res)} OK={len(ok)} cocokDB={len(match)} bedaDB={len(ok) - len(match)} "
      f"ERROR={sum(r.get('status') == 'ERROR' for r in res)} "
      f"median={secs[len(secs) // 2] if secs else None}s mean={round(sum(secs) / max(1, len(secs)), 1)}s")
print("beda DB:", [(r["id"], r["extracted"], r["db_ddmmyy"], r["black"]) for r in ok if r["extracted"] != r["db_ddmmyy"]])
