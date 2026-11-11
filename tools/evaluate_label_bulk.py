"""
Evaluasi akurasi + kecepatan field_type `label_bulk` (LLM Ollama) terhadap kunci
jawaban. Per field: exact match (persis, tanpa normalisasi). Dilaporkan terpisah:
- CORRECT      : terbaca & sama persis
- WRONG_CONF   : lolos gerbang kepercayaan tapi nilai SALAH  <- yang berbahaya (potensi false PASS)
- WRONG_FLAGGED: nilai salah tapi legible=false (aman, masuk REVIEW)
- NULL         : null (aman, masuk REVIEW)

Run: venv\Scripts\python.exe tools\evaluate_label_bulk.py [--truth FILE] [--repeat N]
     [--think false|low|medium] [--limit N] [--dump out.json]
Env: OLLAMA_URL / OLLAMA_MODEL / OLLAMA_THINK dst (lihat llm_extract.py).
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import llm_extract

FIELDS = (("no_batch", "no_batch_raw"), ("mixing_date", "mixing_date_raw"), ("kode_bulk", "kode_bulk"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(ROOT_DIR / "tools" / "eval_data" / "label_bulk_truth.json"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--think", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    if args.think is not None:
        os.environ["OLLAMA_THINK"] = args.think

    items = json.loads(Path(args.truth).read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    cfg = llm_extract._cfg()
    print(f"model={cfg['model']} think={cfg['think']} url={cfg['url']} views={llm_extract._view_sides()} "
          f"items={len(items)} repeat={args.repeat}\n")

    counts = {f: {"CORRECT": 0, "WRONG_CONF": 0, "WRONG_FLAGGED": 0, "NULL": 0} for f, _ in FIELDS}
    times, rows = [], []
    for item in items:
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT_DIR / path
        img = cv2.imread(str(path))
        if img is None:
            print(f"READ_FAILED {path}")
            continue
        for run in range(args.repeat):
            res = llm_extract.analyze_label_bulk(img)
            rows.append({"file": path.name, "run": run, "result": res})
            if res["status"] != "ERROR":
                times.append(res["elapsed_sec"])
            if res["status"] == "ERROR":
                print(f"{path.name:22s} run{run} ERROR {res['error']}")
                continue
            line = []
            for truth_key, res_key in FIELDS:
                if truth_key not in item:  # field tak dinilai
                    continue
                got, want = res[res_key], item[truth_key]
                # "leg" = lolos gerbang kepercayaan (legible + format + semua view sepakat)
                leg = res["legible"][truth_key] and res["format_ok"][truth_key] and res["view_agreement"][truth_key]
                if want is None:  # kunci: harus tidak terbaca (null) - mis. glare/terpotong
                    cat = "CORRECT" if got is None else ("WRONG_CONF" if leg else "WRONG_FLAGGED")
                elif got is None:
                    cat = "NULL"
                elif got == want:
                    cat = "CORRECT"
                else:
                    cat = "WRONG_CONF" if leg else "WRONG_FLAGGED"
                counts[truth_key][cat] += 1
                line.append(f"{truth_key}={got!r}[{cat}]")
            print(f"{path.name:22s} run{run} {res['elapsed_sec']:6.1f}s {res['status']:14s} " + " ".join(line) + f" agree={ {k: v for k, v in res['view_agreement'].items()} }")

    n = sum(counts["no_batch"].values()) or 1
    print("\n=== Akurasi per field (exact match) ===")
    for f, _ in FIELDS:
        c = counts[f]
        tot = sum(c.values()) or 1
        print(f"{f:12s} correct {c['CORRECT']}/{tot} ({100*c['CORRECT']/tot:.0f}%)  "
              f"WRONG_CONFIDENT={c['WRONG_CONF']}  wrong_flagged={c['WRONG_FLAGGED']}  null={c['NULL']}")
    if times:
        print(f"\n=== Kecepatan per foto (detik, hanya request sukses; run pertama termasuk load model) ===\n"
              f"n={len(times)} min={min(times):.1f} median={statistics.median(times):.1f} "
              f"mean={statistics.mean(times):.1f} max={max(times):.1f}")
    if args.dump:
        Path(args.dump).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
