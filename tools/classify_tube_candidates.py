"""Saring foto coding (packing/startup) yang berisi TUBE, via Qwen di Ollama (GPU).

Satu view 1024px, hanya klasifikasi (bukan baca nilai). Hasil = penyaring awal,
tetap dicek mata lewat contact sheet. Resumable. Output:
testdata_private/tube/classify.json  (gitignored)
"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import llm_extract as L  # noqa: E402

SRC = Path(r"C:\Users\cosmax\Downloads")
OUT = ROOT / "testdata_private" / "tube" / "classify.json"
MYSQL = next(Path(r"C:\laragon\bin\mysql").glob("*/bin/mysql.exe"))
PROMPT = """Look at the main product container in this photo of a cosmetics factory.
- container: "tube" = squeezable flexible tube with a flat pressed (crimp-sealed) end; "non_tube" = any other container (box, jar, bottle, cap, stick, compact, pump, pouch); "none" = no product container visible.
- crimp_visible: true only if the flat sealed end of a tube is visible in the photo.
- code_on_crimp: true only if embossed/stamped characters can be seen on that sealed end.
- black_mark_over_code: true only if a dark/black vertical bar or rectangle sits on or touching the code on the sealed end.
- printed_code_on_body: true if a printed (inkjet) code with MFD/EXP/batch is visible on the tube body.
JSON only."""
B = {"type": "boolean"}
SCHEMA = {"type": "object", "properties": {
    "container": {"type": "string", "enum": ["tube", "non_tube", "none"]},
    "crimp_visible": B, "code_on_crimp": B, "black_mark_over_code": B, "printed_code_on_body": B},
    "required": ["container", "crimp_visible", "code_on_crimp", "black_mark_over_code", "printed_code_on_body"]}
Q = ("select a.id, a.stage, a.field_label, a.file_path, b.no_batch, b.exp_date, p.product_name "
     "from ipc_attachments a join ipc_batches b on b.id=a.ipc_batch_id "
     "join master_products p on p.id=b.master_product_id "
     "where a.field_label in ('primary_coding_batch_exp','coding_primary') order by a.id")
rows = subprocess.run([str(MYSQL), "-uroot", "ipc_system", "-B", "-N", "-e", Q], capture_output=True,
                      text=True, encoding="utf-8", errors="replace").stdout.splitlines()
items = []
for r in rows:
    i, st, fl, fp, nb, ex, nm = r.split("\t")
    items.append(dict(id=int(i), stage=st, label=fl, path=str(SRC / fp), db_no_batch=nb, db_exp=ex, product=nm))
# Hasil ditulis ke classify_v2.json (classify.json DIBACA saja, jadi eksperimen yang
# memakainya tidak terganggu). Entri berisi "error" (mis. unreadable karena file waktu itu
# belum ada, atau terowongan putus) dianggap belum selesai dan dicoba ulang.
OLD = OUT
OUT = OLD.with_name("classify_v2.json")
done = {}
for _src in (OLD, OUT):
    if _src.exists():
        for x in json.loads(_src.read_text(encoding="utf-8")):
            if not x.get("error"):
                done[x["id"]] = x
cfg = L._cfg()

def work(it):
    if it["id"] in done:
        return done[it["id"]]
    img = cv2.imread(it["path"])
    if img is None:
        return {**it, "error": "unreadable"}
    try:
        body = L._call_ollama(PROMPT, L._encode_image(img, 1024), SCHEMA, cfg)
        d = L._parse_json(body["message"]["content"])
        return {**it, **d}
    except Exception as e:
        return {**it, "error": f"{type(e).__name__}: {e}"[:120]}

res, n = [], 0
with ThreadPoolExecutor(2) as ex:
    for r in ex.map(work, items):
        res.append(r); n += 1
        if n % 25 == 0:
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"); print(n, len(items), flush=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", len(res), "tube:", sum(r.get("container") == "tube" for r in res))
