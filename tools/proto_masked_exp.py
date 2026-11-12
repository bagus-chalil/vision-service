"""PROTOTYPE mode verifikasi-topeng EXP tube. Model membaca EXP 6 karakter dgn '?' utk
karakter tertutup (garis hitam/glare) TANPA diberi nilai DB; per-posisi konsensus antar
ukuran view (beda -> '?'); pencocokan ke EXP diharapkan dilakukan di kode (wildcard).
Output: testdata_private/tube/masked_exp.json (gitignored)."""
import json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2, llm_extract as L  # noqa: E402
D = ROOT / "testdata_private" / "tube"
OUT = D / "masked_exp.json"
SIDES = [1024, 1600, 2048]
PROMPT = """This photo shows a cosmetic tube. On its flat sealed (crimp) end there is an embossed/stamped line "EXP" followed by a 6-digit expiry date DDMMYY.
Transcribe those 6 characters. A black vertical bar (sensor mark), glare, blur or cut-off may hide some characters.
Rules:
- Output exactly 6 characters: a digit 0-9 for each character you can read with certainty, and "?" for each character that is hidden, ambiguous or unreadable. Keep the left-to-right position of every character (a hidden character is "?" at ITS position, never remove it).
- Never guess or complete a character. "?" is always better than a wrong digit.
- If no "EXP" line with a date is visible, set found=false and exp="??????".
JSON only: {"found": boolean, "exp": string}"""
SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}, "exp": {"type": "string"}}, "required": ["found", "exp"]}

def read_views(img, cfg):
    outs = []
    for s in SIDES:
        try:
            b = L._call_ollama(PROMPT, L._encode_image(img, s), SCHEMA, cfg)
            d = L._parse_json(b["message"]["content"])
            e = d["exp"].replace(" ", "")
            outs.append(e if d["found"] and re.fullmatch(r"[0-9?]{6}", e) else "??????")
        except Exception:
            outs.append("??????")
    return outs

def merge(views):
    # posisi dipercaya hanya bila SEMUA view memberi digit yang sama; selain itu '?'
    return "".join(c[0] if len(set(c)) == 1 and c[0] != "?" else "?" for c in zip(*views))

def compare(merged, db):
    known = [(m, d) for m, d in zip(merged, db) if m != "?"]
    if len(known) < 3: return "NO_SIGNAL"
    if any(m != d for m, d in known): return "CONFLICT"
    return "FULL_MATCH" if len(known) == 6 else "PARTIAL_CONSISTENT"

cfg = L._cfg()
items = [x for x in json.loads((D / "baseline_exp.json").read_text(encoding="utf-8")) if "error" not in x]
done = {x["id"]: x for x in json.loads(OUT.read_text(encoding="utf-8"))} if OUT.exists() else {}
def work(it):
    if it["id"] in done: return done[it["id"]]
    img = cv2.imread(it["path"]); v = read_views(img, cfg); m = merge(v)
    return {"id": it["id"], "path": it["path"], "black_mark": it["black_mark_over_code"], "db": it["db_ddmmyy"],
            "views": v, "merged": m, "verdict": compare(m, it["db_ddmmyy"])}
res, n = [], 0
with ThreadPoolExecutor(2) as ex:
    for r in ex.map(work, items):
        res.append(r); n += 1
        if n % 20 == 0:
            OUT.write_text(json.dumps(res, indent=1), encoding="utf-8"); print(n, len(items), flush=True)
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8"); print("done", len(res))
