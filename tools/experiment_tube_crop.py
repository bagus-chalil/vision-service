"""Eksperimen B: YOLO crop tube -> Qwen (Ollama) baca EXP, vs baseline A (foto utuh).

Hanya YOLO (ringan) yang jalan lokal; TIDAK ada PaddleOCR. Pembacaan memakai
konsensus 4 ukuran yang sama dengan coding_print, TANPA langkah rescue, supaya
satu-satunya variabel yang berbeda dari baseline adalah masukan gambarnya.

Varian crop (dari bbox tube YOLO):
  crimp = 15% atas tube (CRIMP_CROP_TOP_FRACTION pipeline lama)
  top30 = 30% atas tube (lebih longgar, tahan framing)
  tube  = seluruh bbox tube

Kunci = exp_date DB (bukan bacaan foto): OK beda DB = CEK_VISUAL, bukan otomatis salah.
Input : testdata_private/tube/classify.json (175 foto tube berkode crimp)
Output: testdata_private/tube/crop_exp_<varian>.json (gitignored, resumable)
Pakai : venv\\Scripts\\python.exe tools/experiment_tube_crop.py --variant top30 [--limit N]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import llm_extract as L  # noqa: E402
import tube_emboss_pipeline as T  # noqa: E402

D = ROOT / "testdata_private" / "tube"
FRACTION = {"crimp": 0.15, "top30": 0.30, "tube": 1.0}

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=FRACTION, required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=2)
args = ap.parse_args()

OUT = D / f"crop_exp_{args.variant}.json"
items = [x for x in json.loads((D / "classify.json").read_text(encoding="utf-8"))
         if x.get("container") == "tube" and x.get("code_on_crimp")]
if args.limit:
    items = items[: args.limit]
# Entri ERROR (mis. terowongan Ollama putus) TIDAK dianggap selesai: dikerjakan ulang saat dilanjutkan.
done = ({x["id"]: x for x in json.loads(OUT.read_text(encoding="utf-8")) if x.get("status") != "ERROR"}
        if OUT.exists() else {})

# classify.json menyimpan path asli di Downloads\ipc-attachments yang sudah tidak
# lengkap; salinannya ada di photos_candidates (pri_<id>.jpg / cod_<id>.jpg).
_copies = {}
for _p in sorted((D / "photos_candidates").iterdir()):
    _copies.setdefault(_p.stem.split("_", 1)[-1], []).append(_p)
_dupe = [k for k, v in _copies.items() if len(v) > 1]
if _dupe:
    print("PERINGATAN: id ganda di photos_candidates:", _dupe[:5], flush=True)


def resolve_path(it):
    if Path(it["path"]).exists():
        return it["path"]
    c = _copies.get(str(it["id"]))
    return str(c[0]) if c else it["path"]


yolo = T.load_yolo_model()
cfg = L._cfg()


def work(it):
    if it["id"] in done:
        return done[it["id"]]
    base = {k: it.get(k) for k in ("id", "path", "db_exp", "black_mark_over_code")}
    y, m, d = it["db_exp"].split("-")
    base["db_ddmmyy"] = d + m + y[2:]
    img = cv2.imread(resolve_path(it))
    if img is None:
        return {**base, "status": "ERROR", "error": "unreadable"}
    loc = T.localize_tube(img, yolo)  # YOLO dipanggil dari 1 thread per item; model dibagi
    if not loc.get("detected"):
        return {**base, "status": "NO_TUBE", "reason": loc.get("reason")}
    tube = loc["tube_crop"]
    crop = tube[: max(1, int(tube.shape[0] * FRACTION[args.variant])), :]
    try:
        views, _, _ = L._read_coding_views(crop, cfg)
        r = L._coding_consensus(views)
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "ERROR", "error": f"{type(e).__name__}: {e}"[:100]}
    ok = all(r["trusted"][k] for k in ("exp",))
    return {**base, "status": "OK" if ok else "LOW_CONFIDENCE", "exp_raw": r["exp_raw"],
            "exp_format": r["exp_format"], "view_agreement": r["view_agreement"],
            "views_exp": [v["exp"] for v in r["views_read"]],
            "crop_hw": list(crop.shape[:2]), "yolo_conf": loc["confidence"]}


res, n = [], 0
with ThreadPoolExecutor(args.workers) as ex:
    for r in ex.map(work, items):
        res.append(r)
        n += 1
        if n % 10 == 0:
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(n, len(items), flush=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", len(res))
