"""Ekstrak dataset tube ke testdata_private/tube/ (terpisah dari coding_print).

Sumber: (1) DB ipc_system, field_label='image_tube' (SELECT saja) -> file di
Downloads/ipc-attachments; (2) images/ (fixture repo). Output:
  testdata_private/tube/photos/db_<attachment_id>.jpg
  testdata_private/tube/photos/fixture_<nama>
  testdata_private/tube/manifest.json  (kunci DB: batch, mixing, exp, produk)
Folder GITIGNORED - jangan di-commit.
"""
import json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"C:\Users\cosmax\Downloads")
OUT = ROOT / "testdata_private" / "tube"
PH = OUT / "photos"
MYSQL = next(Path(r"C:\laragon\bin\mysql").glob("*/bin/mysql.exe"))

Q = ("select a.id, a.ipc_batch_id, a.file_path, b.no_batch, b.mixing_date, b.exp_date, "
     "p.fg_code, p.product_name from ipc_attachments a "
     "join ipc_batches b on b.id=a.ipc_batch_id "
     "join master_products p on p.id=b.master_product_id "
     "where a.field_label='image_tube' order by a.id")
raw = subprocess.run([str(MYSQL), "-uroot", "ipc_system", "-B", "-N", "-e", Q],
                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
PH.mkdir(parents=True, exist_ok=True)
items, missing = [], 0
for line in raw.splitlines():
    i, bid, fp, nb, mx, ex, fg, name = line.split("\t")
    src = SRC / fp
    if not src.exists():
        missing += 1
        continue
    dst = PH / f"db_{i}{src.suffix.lower()}"
    shutil.copy2(src, dst)
    items.append(dict(file=dst.name, source="db", attachment_id=int(i), batch_id=int(bid),
                      db_no_batch=nb, db_mixing=mx, db_exp=ex, fg_code=fg, product=name))
for f in sorted((ROOT / "images").iterdir()):
    if f.is_file():
        dst = PH / f"fixture_{f.name.replace(' ', '_')}"
        shutil.copy2(f, dst)
        items.append(dict(file=dst.name, source="fixture", original=f.name))
(OUT / "manifest.json").write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"db={sum(x['source']=='db' for x in items)} fixture={sum(x['source']=='fixture' for x in items)} missing={missing}")
