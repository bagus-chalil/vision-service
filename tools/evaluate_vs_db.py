"""
Evaluasi label_bulk / im_document pada SAMPEL ACAK foto produksi, dibandingkan
dengan data DB `ipc_system` (SELECT saja). Dev-only, bukan bagian service.

PENTING - DB BUKAN kunci jawaban mutlak: nilai DB diketik operator, foto bisa
memang beda. Maka hasil dikelompokkan:
  OK_COCOK_DB       : status OK dan semua field sama dengan DB
  OK_BEDA_DB        : status OK tapi ada field beda DB  -> WAJIB cek mata
  LOW / ERROR       : tidak lolos gerbang (aman, masuk REVIEW)
Untuk label_bulk: no_batch, mixing_date (DD/MM/YYYY vs DATE DB), kode_bulk.
Untuk im_document: DB tidak punya nomor IM / umur simpan -> hanya ringkasan
status + kemiripan token versi ("V4") dengan product_name sebagai sanity check.
Perbandingan tanggal di sini hanya untuk evaluasi; service tidak pernah menghitung.

Run: venv\\Scripts\\python.exe tools\\evaluate_vs_db.py --kind label_bulk --n 150 --out FILE.json
"""

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import llm_extract

MYSQL = r"C:\laragon\bin\mysql\mysql-8.4.3-winx64\bin\mysql.exe"
KINDS = {
    "label_bulk": ("date_bulk", llm_extract.analyze_label_bulk),
    "im": ("im_number", llm_extract.analyze_im_document),
}


def db_rows(label):
    sql = (
        "SELECT a.id, b.no_batch, b.mixing_date, b.bulk_code, p.product_name "
        "FROM ipc_attachments a JOIN ipc_batches b ON b.id=a.ipc_batch_id "
        "LEFT JOIN master_products p ON p.id=b.master_product_id "
        f"WHERE a.field_label='{label}'"
    )
    out = subprocess.run([MYSQL, "-uroot", "--default-character-set=utf8mb4", "ipc_system", "-N", "-e", sql],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    rows = {}
    for line in out.splitlines():
        p = line.split("\t")
        rows[int(p[0])] = {"no_batch": p[1], "mixing_date": p[2], "bulk_code": p[3], "product": p[4]}
    return rows


def used_ids():
    ids = set()
    for f in ("label_bulk_truth.json", "label_bulk_truth_v2.json", "im_truth.json"):
        for base in (ROOT_DIR / "testdata_private", ROOT_DIR / "tools" / "eval_data"):
            fp = base / f
            if fp.exists():
                for it in json.loads(fp.read_text(encoding="utf-8"))["items"]:
                    m = re.search(r"(\d+)\.jpg$", it["path"])
                    if m:
                        ids.add(int(m.group(1)))
    return ids


def ddmmyyyy(d):  # '2026-10-08' -> '08/10/2026' (hanya untuk perbandingan evaluasi)
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", d or "")
    return f"{m.group(3)}/{m.group(2)}/{m.group(1)}" if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    folder, fn = KINDS[args.kind]
    db = db_rows(folder)
    skip = used_ids()
    files = sorted(int(p.stem) for p in (ROOT_DIR / "testdata_private" / folder).glob("*.jpg"))
    pool = [i for i in files if i not in skip and i in db]
    random.Random(args.seed).shuffle(pool)
    sample = pool[: args.n]
    print(f"kind={args.kind} pool={len(pool)} sample={len(sample)} (di luar {len(skip)} id yang sudah pernah diukur)", flush=True)

    results = []
    for k, i in enumerate(sample, 1):
        img = cv2.imread(str(ROOT_DIR / "testdata_private" / folder / f"{i}.jpg"))
        if img is None:
            continue
        t0 = time.perf_counter()
        res = fn(img)
        row = {"id": i, "db": db[i], "result": res, "sec": round(time.perf_counter() - t0, 1)}
        results.append(row)
        if k % 10 == 0:
            Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{k}/{len(sample)} status so far: "
                  f"{sum(1 for r in results if r['result']['status']=='OK')} OK", flush=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("selesai", len(results), flush=True)


if __name__ == "__main__":
    main()
