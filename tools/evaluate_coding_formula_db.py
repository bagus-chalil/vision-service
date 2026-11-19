"""Uji parser formula coding ke isian operator di DB (startup_checks.coding_format_*), tanpa GPU.

Untuk tiap isian: (1) usulkan formula dari contoh itu (coding_formula.formula_from_example,
exp_date DB hanya untuk memilih tafsiran 6 digit), (2) urai contoh itu sendiri dengan formula
tsb, (3) bandingkan batch/EXP hasil urai dengan ipc_batches (no_batch, exp_date).
Tujuannya menguji CAKUPAN token + parser pada format nyata, bukan akurasi AI.

Kategori:
  NO_FORMULA   formula tidak bisa diusulkan (contoh tak dikenali / perlu konfirmasi QC)
  PARSE_FAIL   formula ada tapi batch/EXP contoh itu sendiri tidak FOUND -> celah parser
  MATCH_DB     batch & EXP cocok DB (EXP dibanding sesuai granularity: hari atau bulan)
  DIFF_DB      terurai tapi beda DB (bisa coding memang beda, operator salah ketik, dsb.)

Jalankan: venv\\Scripts\\python.exe tools\\evaluate_coding_formula_db.py [--show N]
Output JSON: testdata_private/runs/coding_formula/db_eval.json
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import coding_formula as cf  # noqa: E402
from dataset import RUNS  # noqa: E402

MYSQL = next(Path(r"C:\laragon\bin\mysql").glob("*/bin/mysql.exe"))
SKIP = {"", "N/A", "NA", "-", "TIDAK ADA"}


def sql(q):
    r = subprocess.run([str(MYSQL), "-uroot", "ipc_system", "-B", "-N", "-e", q],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        sys.exit(r.stderr)
    return [[None if v == "NULL" else v for v in l.split("\t")] for l in r.stdout.splitlines()]


def db_batch_letters(no_batch):
    """'26Y070/ULY' / '26X005 EKB' -> 'ULY' / 'EKB' (3 huruf setelah / atau spasi)."""
    if not no_batch:
        return None
    for sep in ("/", " "):
        if sep in no_batch:
            tail = no_batch.split(sep)[-1].strip()
            return tail or None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=15, help="contoh per kategori yang dicetak")
    args = ap.parse_args()

    rows = sql("""SELECT s.id, x.level, x.txt, b.no_batch, b.exp_date
      FROM startup_checks s JOIN ipc_batches b ON b.id = s.ipc_batch_id
      JOIN (SELECT id, 'primary' level, coding_format_primary txt FROM startup_checks
            UNION ALL SELECT id, 'secondary', coding_format_secondary FROM startup_checks
            UNION ALL SELECT id, 'tersier', coding_format_tersier FROM startup_checks) x ON x.id = s.id
      WHERE x.txt IS NOT NULL AND b.deleted_at IS NULL""")
    results, cats = [], Counter()
    for sid, level, txt, no_batch, exp_date in rows:
        if txt.strip().upper() in SKIP:
            continue
        prop = cf.formula_from_example(txt, exp_date=exp_date)
        rec = {"startup_check_id": int(sid), "level": level, "text": txt, "db_no_batch": no_batch,
               "db_exp_date": exp_date, "formula": prop["segments"], "notes": prop["notes"]}
        if not prop["segments"]:
            rec["category"] = "NO_FORMULA"
        else:
            out = cf.extract(txt.replace("\\n", "\n").split("\n"), prop["segments"])
            fb, fe = out["fields"].get("batch"), out["fields"].get("exp")
            rec["parsed"] = {k: {"status": v["status"], "value": v.get("value"), "date": v.get("date")}
                             for k, v in out["fields"].items()}
            if not fe or fe["status"] != "FOUND" or (fb and fb["status"] != "FOUND"):
                rec["category"] = "PARSE_FAIL"
            else:
                diffs = []
                want_b = db_batch_letters(no_batch)
                if fb and want_b and fb["value"] != want_b:
                    diffs.append(f"batch {fb['value']} vs DB {want_b}")
                if exp_date:
                    y, m, d = int(exp_date[:4]), int(exp_date[5:7]), int(exp_date[8:10])
                    got = fe["date"]
                    if (got["year"], got["month"]) != (y, m) or (got["day"] is not None and got["day"] != d):
                        diffs.append(f"exp {fe['value']} vs DB {exp_date}")
                if not fb:
                    diffs.append("formula tanpa batch")
                rec["diffs"] = diffs
                rec["category"] = "DIFF_DB" if diffs else "MATCH_DB"
        cats[rec["category"]] += 1
        results.append(rec)

    total = len(results)
    print(f"{total} isian coding (non-kosong) dari DB")
    for k in ("MATCH_DB", "DIFF_DB", "PARSE_FAIL", "NO_FORMULA"):
        print(f"  {k:<11} {cats[k]:>4}  ({cats[k] / max(total, 1):.0%})")
    gran = Counter(r["parsed"]["exp"]["date"]["day"] is None for r in results
                   if r.get("parsed", {}).get("exp", {}).get("date"))
    print(f"  EXP granularity: hari {gran[False]}, bulan {gran[True]}")
    seg_shapes = Counter(" | ".join(r["formula"]) for r in results if r["formula"])
    print("\nFormula terbanyak:")
    for s, n in seg_shapes.most_common(12):
        print(f"  {n:>4}  {s}")
    for k in ("PARSE_FAIL", "NO_FORMULA", "DIFF_DB"):
        sub = [r for r in results if r["category"] == k]
        if sub:
            print(f"\n--- {k} (contoh {min(args.show, len(sub))} dari {len(sub)})")
            for r in sub[: args.show]:
                extra = r.get("diffs") or r.get("notes") or r.get("parsed")
                print(f"  {r['text'][:60]!r:<64} -> {r['formula']} {extra}")

    out_dir = RUNS / "coding_formula"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "db_eval.json").write_text(json.dumps(
        {"summary": dict(cats), "total": total, "results": results}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print(f"\n-> {out_dir / 'db_eval.json'}")


if __name__ == "__main__":
    main()
