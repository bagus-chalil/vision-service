"""Tuning baca EXP tube: crop (YOLO, Python) -> baca (Ollama) -> konsensus (Python).

Pembagian tugas yang diuji:
  Python : YOLO lokalisasi + crop, validasi format DDMMYY, konsensus semua view
  Ollama : hanya TRANSKRIPSI teks pada crop (tanpa keputusan)
Opsi yang diukur (akurasi DAN kecepatan):
  --crop     fraksi atas bbox tube (0.30 = top30; 0 = foto utuh, tanpa YOLO)
  --prompt   general (CODING_PROMPT lama) | crimp (prompt khusus crimp emboss)
  --early    berhenti begitu satu view tidak sepakat (hasil 'trusted' identik,
             karena konsensus butuh SEMUA view sama; hanya menghemat panggilan)
  --parallel view dibaca paralel
Kunci = exp_date DB: OK beda DB = CEK_VISUAL (cetakan bisa memang beda dari DB).
Set: old = 175 foto classify.json | new = foto tube berkode crimp BARU di classify_v2.json
(hold-out: jangan dipakai untuk menyetel).
Output: testdata_private/tube/tune_<nama>.json (gitignored).
"""
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

D = ROOT / "testdata_private" / "tube"

CRIMP_PROMPT = """This is a photo (or close-up crop) of the flat sealed end (crimp) of a cosmetics tube. Characters are EMBOSSED (raised, low contrast, same color as the tube), not printed. The line normally reads "EXP" followed by a 6-digit date DDMMYY, e.g. "EXP 211029" (21 October 2029). Other short codes may sit on the same or other lines - ignore them.

Transcribe ONLY the 6 characters after "EXP", exactly as embossed, character by character. Do not correct, complete or guess.
- A black bar/rectangle, glare, blur, a cut-off edge or anything covering even ONE character makes the whole value unreadable: return null and legible=false. A wrong confident value is far worse than null.
- legible is true only if you can see every one of the 6 characters with certainty.
- If there is no "EXP" embossed line, return null and legible=false.

JSON only: {"exp": string|null, "legible": boolean}"""
CRIMP_SCHEMA = {"type": "object",
                "properties": {"exp": {"type": ["string", "null"]}, "legible": {"type": "boolean"}},
                "required": ["exp", "legible"]}


def _validate_crimp(d):
    if not (d.get("exp") is None or isinstance(d.get("exp"), str)) or not isinstance(d.get("legible"), bool):
        raise L.LlmBadJson("crimp view malformed")


ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--set", choices=("old", "new"), default="old")
ap.add_argument("--crop", type=float, default=0.30)
ap.add_argument("--prompt", choices=("general", "crimp"), default="general")
ap.add_argument("--early", action="store_true")
ap.add_argument("--parallel", action="store_true")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--sides", default=None, help="mis. 1024,1600 (default CODING_VIEW_SIDES)")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

OUT = D / f"tune_{args.name}.json"
old_ids = {x["id"] for x in json.loads((D / "classify.json").read_text(encoding="utf-8"))}
allc = json.loads((D / "classify_v2.json").read_text(encoding="utf-8"))
items = [x for x in allc if x.get("container") == "tube" and x.get("code_on_crimp") and not x.get("error")
         and x.get("db_exp") not in (None, "", "NULL")
         and ((x["id"] in old_ids) if args.set == "old" else (x["id"] not in old_ids))]
if args.limit:
    items = items[: args.limit]
done = ({x["id"]: x for x in json.loads(OUT.read_text(encoding="utf-8")) if x.get("status") != "ERROR"}
        if OUT.exists() else {})
yolo = None
if args.crop > 0:
    import tube_emboss_pipeline as T
    yolo = T.load_yolo_model()
cfg = L._cfg()
sides = [int(s) for s in args.sides.split(",")] if args.sides else L._view_sides("CODING_VIEW_SIDES")


def read_view(crop, side):
    b64 = L._encode_image(crop, side)
    if args.prompt == "crimp":
        d, _, _ = L._extract_with_retry(CRIMP_PROMPT, b64, CRIMP_SCHEMA, cfg, _validate_crimp)
        return {"exp": d["exp"], "legible": d["legible"]}
    d, _, _ = L._extract_with_retry(L.CODING_PROMPT, b64, L.CODING_SCHEMA, cfg, L._validate_coding)
    return {"exp": d["exp"], "legible": bool(d["legible"]["exp"])}


def norm(v):
    return v.strip() if (v is not None and v.strip()) else None


def work(it):
    if it["id"] in done:
        return done[it["id"]]
    y, m, d = it["db_exp"].split("-")
    base = {"id": it["id"], "db_ddmmyy": d + m + y[2:], "black": bool(it.get("black_mark_over_code"))}
    t0 = time.perf_counter()
    img = cv2.imread(it["path"])
    if img is None:
        return {**base, "status": "ERROR", "error": "unreadable"}
    crop = img
    if args.crop > 0:
        loc = T.localize_tube(img, yolo)
        if not loc.get("detected"):
            return {**base, "status": "NO_TUBE", "sec": round(time.perf_counter() - t0, 2)}
        tube = loc["tube_crop"]
        crop = tube[: max(1, int(tube.shape[0] * args.crop)), :] if args.crop < 1 else tube
    views = []
    try:
        if args.parallel and not args.early:
            with ThreadPoolExecutor(len(sides)) as ex:
                views = list(ex.map(lambda s: read_view(crop, s), sides))
        else:
            for s in sides:
                v = read_view(crop, s)
                views.append(v)
                if args.early and (norm(v["exp"]) is None or not v["legible"]
                                   or norm(v["exp"]) != norm(views[0]["exp"])):
                    break
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "ERROR", "error": f"{type(e).__name__}: {e}"[:100]}
    first = norm(views[0]["exp"])
    trusted = (len(views) == len(sides) and first is not None
               and bool(L.CODING_EXP_DDMMYY.match(first))
               and all(norm(v["exp"]) == first and v["legible"] for v in views))
    return {**base, "status": "OK" if trusted else "LOW_CONFIDENCE", "exp_raw": first,
            "views_exp": [v["exp"] for v in views], "n_views": len(views),
            "sec": round(time.perf_counter() - t0, 2)}


res = []
t_all = time.perf_counter()
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
print(f"[{args.name}] n={len(res)} OK={len(ok)} cocokDB={len(match)} bedaDB={len(ok) - len(match)} "
      f"NO_TUBE={sum(r['status'] == 'NO_TUBE' for r in res)} ERROR={sum(r['status'] == 'ERROR' for r in res)} "
      f"median={secs[len(secs) // 2] if secs else None}s total={round(time.perf_counter() - t_all)}s")
