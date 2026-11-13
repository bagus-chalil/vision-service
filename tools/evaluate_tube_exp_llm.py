"""Evaluasi field_type tube_exp_llm (fungsi final, bukan simulasi) vs kunci DB.

Memanggil persis yang dipanggil main.py: tube_emboss_pipeline.tube_exp_crops (YOLO)
-> llm_extract.analyze_tube_exp (Ollama + aturan 3-dari-4). Tanpa PaddleOCR, tanpa
import main. Set: old = 175 foto tube berkode crimp (dipakai menyetel), new = hold-out.
Kunci = exp_date DB: OK beda DB = CEK_VISUAL (cetakan bisa memang beda dari DB).
Pakai: venv\\Scripts\\python.exe tools/evaluate_tube_exp_llm.py --set old|new
Output: testdata_private/tube/final_<set>.json (gitignored, resumable)."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import llm_extract as L  # noqa: E402
import tube_emboss_pipeline as T  # noqa: E402

D = ROOT / "testdata_private" / "tube"
ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=("old", "new"), required=True)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

OUT = D / f"final_{args.set}.json"
old_ids = {x["id"] for x in json.loads((D / "classify.json").read_text(encoding="utf-8"))}
items = [x for x in json.loads((D / "classify_v2.json").read_text(encoding="utf-8"))
         if x.get("container") == "tube" and x.get("code_on_crimp") and not x.get("error")
         and x.get("db_exp") not in (None, "", "NULL")
         and ((x["id"] in old_ids) if args.set == "old" else (x["id"] not in old_ids))]
if args.limit:
    items = items[: args.limit]
done = ({x["id"]: x for x in json.loads(OUT.read_text(encoding="utf-8")) if x.get("status") != "ERROR"}
        if OUT.exists() else {})
yolo = T.load_yolo_model()


def work(it):
    if it["id"] in done:
        return done[it["id"]]
    y, m, d = it["db_exp"].split("-")
    base = {"id": it["id"], "db_ddmmyy": d + m + y[2:], "black": bool(it.get("black_mark_over_code"))}
    img = cv2.imread(it["path"])
    if img is None:
        return {**base, "status": "ERROR", "error": "unreadable"}
    t0 = time.perf_counter()
    c = T.tube_exp_crops(img, yolo)
    r = L.analyze_tube_exp(img, c["crops"] if c["detected"] else [])
    return {**base, "status": r["status"], "exp_raw": r.get("exp_raw"), "error": r.get("error"),
            "trusted_sources": r.get("trusted_sources"), "sources_read": r.get("sources_read"),
            "tube_detected": c["detected"], "sec": round(time.perf_counter() - t0, 2),
            "sources": [(s["source"], s["exp"], s["trusted"]) for s in r.get("sources", [])]}


res = []
with ThreadPoolExecutor(args.workers) as ex:
    for i, r in enumerate(ex.map(work, items), 1):
        res.append(r)
        if i % 20 == 0:
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(i, len(items), flush=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
ok = [r for r in res if r["status"] == "OK"]
match = [r for r in ok if r["exp_raw"] == r["db_ddmmyy"]]
secs = sorted(r["sec"] for r in res if "sec" in r)
print(f"[final {args.set}] n={len(res)} OK={len(ok)} cocokDB={len(match)} bedaDB={len(ok) - len(match)} "
      f"ERROR={sum(r['status'] == 'ERROR' for r in res)} median={secs[len(secs) // 2]}s "
      f"mean={round(sum(secs) / len(secs), 1)}s total={round(sum(secs))}s")
print("beda DB:", [(r["id"], r["exp_raw"], r["db_ddmmyy"], r["black"]) for r in ok if r["exp_raw"] != r["db_ddmmyy"]])
