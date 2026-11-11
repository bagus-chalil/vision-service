"""
Evaluasi akurasi + kecepatan field_type `im_document` (LLM Ollama) terhadap kunci
jawaban (default: testdata_private/im_truth.json - privat, gitignored).

Per item, dua field dinilai (exact match, tanpa normalisasi):
- im_number : string persis, atau null (harus tidak terbaca / bukan IM)
- shelf_life: list [{"packaging", "years", "months"}] - dibandingkan sebagai himpunan
              (huruf kecil, spasi dirapikan); [] = tidak ada pernyataan umur simpan terbaca.
Kategori (sama dengan evaluate_label_bulk.py):
- CORRECT       : sama dengan kunci
- WRONG_CONF    : lolos gerbang kepercayaan tapi SALAH  <- berbahaya (potensi false PASS)
- WRONG_FLAGGED : salah tapi tidak lolos gerbang (aman, masuk REVIEW)
- NULL          : kosong padahal kunci berisi (aman, masuk REVIEW)

Run: venv\\Scripts\\python.exe tools\\evaluate_im_document.py [--truth FILE] [--limit N] [--dump out.json]
Env: OLLAMA_URL / OLLAMA_MODEL / OLLAMA_THINK / IM_VIEW_SIDES (lihat llm_extract.py).
Catatan: kunci jawaban harus dibuat dari pembacaan MANUSIA atas foto, bukan dari keluaran model.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import llm_extract


def shelf_set(entries):
    return sorted(
        (re.sub(r"\s+", " ", e["packaging"]).strip().lower(), e.get("years"), e.get("months"))
        for e in entries
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(ROOT_DIR / "testdata_private" / "im_truth.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    items = json.loads(Path(args.truth).read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    cfg = llm_extract._cfg()
    print(f"model={cfg['model']} think={cfg['think']} url={cfg['url']} "
          f"views={llm_extract._view_sides('IM_VIEW_SIDES')} items={len(items)}\n")

    cats = ("CORRECT", "WRONG_CONF", "WRONG_FLAGGED", "NULL")
    counts = {f: dict.fromkeys(cats, 0) for f in ("im_number", "shelf_life")}
    status_counts, times, rows = {}, [], []
    for item in items:
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT_DIR / path
        img = cv2.imread(str(path))
        if img is None:
            print(f"READ_FAILED {path}")
            continue
        res = llm_extract.analyze_im_document(img)
        rows.append({"file": path.name, "result": res})
        status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
        if res["status"] == "ERROR":
            print(f"{path.name:14s} ERROR {res['error']}")
            continue
        times.append(res["elapsed_sec"])
        line = []
        for field in ("im_number", "shelf_life"):
            if field not in item:
                continue
            trusted = res["legible"][field] and res["format_ok"][field] and res["view_agreement"][field]
            if field == "im_number":
                got, want = res["im_number"], item["im_number"]
                same = got == want
                empty = got is None
            else:
                got, want = shelf_set(res["shelf_life"]), shelf_set(item["shelf_life"])
                same = got == want
                empty = not res["shelf_life"]
            if same:
                cat = "CORRECT"
            elif empty:
                cat = "NULL"
            else:
                cat = "WRONG_CONF" if trusted else "WRONG_FLAGGED"
            counts[field][cat] += 1
            line.append(f"{field}[{cat}]")
        print(f"{path.name:14s} {res['elapsed_sec']:5.1f}s {res['status']:14s} " + " ".join(line))

    print("\n=== Akurasi per field (exact match) ===")
    for field, c in counts.items():
        tot = sum(c.values()) or 1
        print(f"{field:11s} correct {c['CORRECT']}/{tot} ({100*c['CORRECT']/tot:.0f}%)  "
              f"WRONG_CONFIDENT={c['WRONG_CONF']}  wrong_flagged={c['WRONG_FLAGGED']}  null={c['NULL']}")
    print(f"\nStatus: {status_counts}")
    if times:
        print(f"Kecepatan per foto (detik): n={len(times)} min={min(times):.1f} "
              f"median={statistics.median(times):.1f} max={max(times):.1f}")
    if args.dump:
        Path(args.dump).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
