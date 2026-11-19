"""Analisis offline hasil tools/evaluate_coding_formula_photos.py (tanpa GPU).

Bacaan model per view sudah tersimpan, jadi tiap varian diurai ulang dari data yang sama:
  - formula: SKU (diusulkan dari isian operator) vs AUTO (gabungan katalog, wajib batch+exp)
  - view: subset/urutan ukuran, dengan early-exit seperti di produksi
Ringkasan per varian: OK, OK cocok DB, OK beda DB (wajib dicek mata), panggilan & detik rata-rata.

Jalankan: venv\\Scripts\\python.exe tools\\analyze_coding_formula_runs.py [--tags holdout train] [--list-diff]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import coding_formula as cf  # noqa: E402
import llm_extract as le  # noqa: E402
from dataset import RUNS  # noqa: E402
from evaluate_coding_formula_photos import db_letters  # noqa: E402

SIDES = [1024, 640, 1600, 2048]
AUTO_SEGS = cf.compile_formula(cf.auto_segments())


def run(entry, sides, auto):
    segs = AUTO_SEGS if auto else cf.compile_formula(entry["formula"])
    wanted = set().union(*(s.fields for s in segs))
    req = [f for f in (("batch", "exp") if auto else cf.REQUIRED_FIELDS) if f in wanted]
    by_side = {v["side"]: v for v in entry["views"]}
    per_view, secs = [], 0.0
    for side in sides:
        pv = cf.extract(by_side[side]["lines"], segs)
        if by_side[side].get("error"):
            pv["fields"] = {f: {"status": "VIEW_FAILED", "value": None} for f in pv["fields"]}
        per_view.append(pv)
        secs += by_side[side]["sec"]
        k0 = {f: cf.key_of(per_view[0]["fields"].get(f)) for f in req}
        if not [f for f in req if k0[f] is not None and cf.key_of(per_view[-1]["fields"].get(f)) == k0[f]]:
            break
    fields = le._formula_consensus(per_view, req)
    ok = bool(req) and all(fields[f]["status"] == "OK" for f in req)
    return ok, fields, len(per_view), secs


def diff_db(entry, fields):
    """Daftar beda dengan DB untuk hasil OK (kosong = cocok)."""
    out = []
    y, m, d = (int(x) for x in entry["db_exp_date"][:10].split("-"))
    e = fields["exp"]["date"]
    if (e["year"], e["month"]) != (y, m) or (e["day"] is not None and e["day"] != d):
        out.append(f"exp {fields['exp']['value']} vs DB {entry['db_exp_date'][:10]}")
    b = fields.get("batch", {}).get("value")
    want = db_letters(entry["db_no_batch"])
    if b and want and len(b) == 3 and b != want:
        out.append(f"batch {b} vs DB {want}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["holdout", "train"])
    ap.add_argument("--list-diff", action="store_true")
    args = ap.parse_args()
    entries = []
    for tag in args.tags:
        entries += json.loads((RUNS / "coding_formula" / f"photos_{tag}.json").read_text(encoding="utf-8"))
    print(f"{len(entries)} foto ({', '.join(args.tags)})\n")
    configs = [SIDES] + [list(c) for c in itertools.combinations(SIDES, 3)] + \
              [list(c) for c in itertools.combinations(SIDES, 2)]
    diffs = {}
    for auto in (False, True):
        print("== formula", "AUTO (katalog)" if auto else "SKU (dari isian operator)")
        for sides in configs:
            n_ok = n_match = 0
            calls, secs, ok_secs, dl = [], [], [], []
            for e in entries:
                ok, fields, n, s = run(e, sides, auto)
                calls.append(n)
                secs.append(s)
                if ok:
                    ok_secs.append(s)
                    n_ok += 1
                    dd = diff_db(e, fields)
                    n_match += not dd
                    if dd:
                        dl.append((e["attachment_id"], dd))
            diffs[(auto, tuple(sides))] = dl
            secs.sort()
            print(f"  {','.join(map(str, sides)):<20} OK {n_ok:>3} ({n_ok / len(entries):.0%})  cocok DB {n_match:>3}  "
                  f"beda DB {n_ok - n_match:>2}  panggilan {sum(calls) / len(calls):.2f}  "
                  f"dtk median {secs[len(secs) // 2]:.1f} rata2 {sum(secs) / len(secs):.1f}"
                  + (f"  kasus OK {sum(ok_secs) / len(ok_secs):.1f}" if ok_secs else ""))
        print()
    if args.list_diff:
        allids = sorted({a for dl in diffs.values() for a, _ in dl})
        print("OK beda DB (union semua varian) -> cek mata:")
        for a in allids:
            where = [("A" if auto else "S") + ":" + ",".join(map(str, s)) for (auto, s), dl in diffs.items()
                     if any(x == a for x, _ in dl)]
            why = next(dd for dl in diffs.values() for x, dd in dl if x == a)
            img = next(e["image"] for e in entries if e["attachment_id"] == a)
            print(f"  {a}  {why}  {img}  di {len(where)} varian: {' '.join(where[:6])}{' ...' if len(where) > 6 else ''}")


if __name__ == "__main__":
    main()
