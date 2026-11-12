"""Baseline baca EXP tube (jalur Ollama coding_print) vs DB, dipecah per garis hitam.
Input: testdata_private/tube/classify.json. Output: testdata_private/tube/baseline_exp.json (gitignored).
Kunci DB = exp_date batch (BUKAN bacaan foto): OK beda DB = CEK_VISUAL, bukan otomatis salah."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2, llm_extract as L  # noqa: E402
D = ROOT / "testdata_private" / "tube"
OUT = D / "baseline_exp.json"
items = [x for x in json.loads((D / "classify.json").read_text(encoding="utf-8"))
         if x.get("container") == "tube" and x.get("code_on_crimp")]
done = {x["id"]: x for x in json.loads(OUT.read_text(encoding="utf-8"))} if OUT.exists() else {}

def work(it):
    if it["id"] in done:
        return done[it["id"]]
    img = cv2.imread(it["path"])
    try:
        r = L.analyze_coding_print(img)
    except Exception as e:
        return {**it, "error": f"{type(e).__name__}: {e}"[:100]}
    y, m, d = it["db_exp"].split("-")
    return {**it, "status": r.get("status"), "exp_raw": r.get("exp_raw"), "batch_raw": r.get("batch_raw"),
            "exp_format": r.get("exp_format"), "db_ddmmyy": d + m + y[2:], "db_mm_yyyy": f"{m} {y}",
            "read_via": r.get("read_via"), "packaging_type": r.get("packaging_type")}
res, n = [], 0
with ThreadPoolExecutor(2) as ex:
    for r in ex.map(work, items):
        res.append(r); n += 1
        if n % 20 == 0:
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"); print(n, len(items), flush=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"); print("done", len(res))
