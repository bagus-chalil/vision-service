"""Evaluasi field_type `coding_print` (batch + EXP coding NON-TUBE) vs kunci jawaban.

Dev-only. Memanggil llm_extract.analyze_coding_print() langsung (kode yang sama
dengan /api/analyze), jadi semua inferensi tetap di server Ollama (GPU) lewat
OLLAMA_URL - tidak ada OCR lokal.

Dua jenis kunci (JSON {"items": [...]}, path relatif ke root repo):
  - kunci bacaan foto  : {"path", "batch", "exp"}   (null = tak ada coding terbaca)
  - kunci data DB batch: {"path", "db_batch", "db_exp_ddmmyy", "db_exp_mm_yyyy"}
    -> DB BUKAN bacaan foto (coding bisa memang beda dari DB). OK yang tidak cocok
       DB dicetak sebagai CEK_VISUAL, bukan otomatis dianggap salah.

Kategori terpenting: WRONG_CONFIDENT (status OK tapi nilai salah = berbahaya).

Contoh:
  venv\\Scripts\\python.exe tools\\evaluate_coding_print.py --truth testdata_private\\coding_print_truth.json
  venv\\Scripts\\python.exe tools\\evaluate_coding_print.py --truth testdata_private\\coding_print\\_set2_db.json --out debug_output\\set2.json

Foto & kunci produksi ada di testdata_private/ (gitignored) - jangan di-commit.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import llm_extract  # noqa: E402


def expected_for(item, result):
    """Return (batch, exp, source) to compare against, or (None, None, src)."""
    if "batch" in item or "exp" in item:
        return item.get("batch"), item.get("exp"), "foto"
    exp = item.get("db_exp_mm_yyyy") if result.get("exp_format") == "MM_YYYY" else item.get("db_exp_ddmmyy")
    return item.get("db_batch"), exp, "db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", help="simpan hasil mentah per foto (JSON)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.load(open(args.truth, encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    counts, rows, times = Counter(), [], []
    for i, item in enumerate(items, 1):
        img = cv2.imread(str(ROOT / item["path"]))
        if img is None:
            print(f"[{i}] GAGAL BACA FILE {item['path']}")
            counts["file_error"] += 1
            continue
        t = time.perf_counter()
        r = llm_extract.analyze_coding_print(img)
        times.append(time.perf_counter() - t)
        exp_b, exp_e, src = expected_for(item, r)
        got = (r.get("batch_raw"), r.get("exp_raw"))
        if r["status"] == "ERROR":
            cat = "ERROR"
        elif r["status"] == "OK":
            if got == (exp_b, exp_e):
                cat = "OK_CORRECT"
            elif src == "db":
                cat = "OK_BEDA_DB_CEK_VISUAL"
            else:
                cat = "WRONG_CONFIDENT"
        else:
            cat = "LOW_ada_coding" if (src == "foto" and exp_b) else "LOW"
        counts[cat] += 1
        if r.get("read_via") == "crop_rescue" and r["status"] == "OK":
            counts["via_crop_rescue"] += 1
        if r.get("exp_format"):
            counts[f"exp_format_{r['exp_format']}_{r['status']}"] += 1
        counts[f"packaging_{r.get('packaging_type')}"] += 1
        print(f"[{i}/{len(items)}] {cat:22s} {Path(item['path']).name:28s} baca={got} harap={(exp_b, exp_e)} "
              f"via={r.get('read_via')} pkg={r.get('packaging_type')} {times[-1]:.1f}s", flush=True)
        rows.append({"item": item, "category": cat, "result": r})
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("\n=== Ringkasan ===")
    for k in sorted(counts):
        print(f"  {k:32s} {counts[k]}")
    if times:
        print(f"  rata-rata detik/foto             {sum(times) / len(times):.1f}")
    if counts["WRONG_CONFIDENT"]:
        print("\n!!! ADA WRONG_CONFIDENT (false OK) - jangan dipakai sebelum diselidiki.")


if __name__ == "__main__":
    main()
