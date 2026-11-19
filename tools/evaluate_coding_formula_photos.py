"""Evaluasi coding_formula (Ollama GPU) di foto coding master + simulasi kecepatan offline.

Per foto: formula diusulkan dari isian coding operator batch itu (startup_checks.coding_format_*,
level sesuai field_label), lalu SEMUA view dibaca sekali (tanpa early exit) dan disimpan beserta
waktunya. Varian view (subset/urutan/early-exit) lalu disimulasikan dari bacaan yang sama, jadi
perbandingan kecepatan vs keamanan tidak butuh panggilan GPU tambahan.

Kunci = DB (exp_date, 3 huruf no_batch). OK yang beda DB WAJIB dicek mata (coding bisa memang beda).

Jalankan: venv\\Scripts\\python.exe tools\\evaluate_coding_formula_photos.py --n 60 [--split eval]
Output: testdata_private/runs/coding_formula/photos_<tag>.json (cache: entri yang sudah ada dilewati)
"""
import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2  # noqa: E402

import coding_formula as cf  # noqa: E402
import llm_extract as le  # noqa: E402
from dataset import MASTER, RUNS  # noqa: E402

MYSQL = next(Path(r"C:\laragon\bin\mysql").glob("*/bin/mysql.exe"))
LEVEL = {"coding": "primary", "coding_primary": "primary", "primary_coding_batch_exp": "primary",
         "coding_secondary": "secondary", "secondary_coding_batch_exp": "secondary"}
SIDES = [1024, 640, 1600, 2048]


def sql(q):
    r = subprocess.run([str(MYSQL), "-uroot", "ipc_system", "-B", "-N", "-e", q],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        sys.exit(r.stderr)
    return [[None if v == "NULL" else v for v in l.split("\t")] for l in r.stdout.splitlines()]


def db_letters(no_batch):
    for sep in ("/", " "):
        if no_batch and sep in no_batch:
            return no_batch.split(sep)[-1].strip() or None
    return None


def pick(n, split, seed):
    typed = {}
    for bid, p, s in sql("SELECT ipc_batch_id, coding_format_primary, coding_format_secondary FROM startup_checks"):
        typed[int(bid)] = {"primary": p, "secondary": s}
    rows = []
    for line in open(MASTER / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["status"] != "usable" or r["field_label"] not in LEVEL or r["split"] != split:
            continue
        if r.get("used_in_keys") or not r.get("db_exp_date") or "batch_deleted" in (r.get("flags") or ""):
            continue
        txt = (typed.get(int(r["batch_id"])) or {}).get(LEVEL[r["field_label"]])
        if not txt or txt.strip().upper() in ("N/A", "-", ""):
            continue
        prop = cf.formula_from_example(txt, exp_date=r["db_exp_date"])
        if not prop["segments"]:
            continue
        rows.append({**r, "typed": txt, "formula": prop["segments"]})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def read_all_views(img, cfg):
    views = []
    for side in SIDES:
        t = time.perf_counter()
        lines, err = le.read_code_lines(img, side, cfg)
        views.append({"side": side, "lines": lines or [], "sec": round(time.perf_counter() - t, 2),
                      **({"error": err} if err else {})})
    return views


def simulate(entry, sides, early_exit=True):
    """Status per field + jumlah panggilan untuk satu konfigurasi view, dari bacaan tersimpan."""
    segs = cf.compile_formula(entry["formula"])
    req = [f for f in cf.REQUIRED_FIELDS if f in set().union(*(s.fields for s in segs))]
    by_side = {v["side"]: v for v in entry["views"]}
    per_view, secs = [], 0.0
    for side in sides:
        pv = cf.extract(by_side[side]["lines"], segs)
        if by_side[side].get("error"):
            pv["fields"] = {f: {"status": "VIEW_FAILED", "value": None} for f in pv["fields"]}
        per_view.append(pv)
        secs += by_side[side]["sec"]
        if early_exit:
            k0 = {f: cf.key_of(per_view[0]["fields"].get(f)) for f in req}
            if not [f for f in req if k0[f] is not None and cf.key_of(per_view[-1]["fields"].get(f)) == k0[f]]:
                break
    fields = le._formula_consensus(per_view, req)
    status = "OK" if req and all(fields[f]["status"] == "OK" for f in req) else "LOW_CONFIDENCE"
    return status, fields, len(per_view), secs


def verdict(entry, status, fields):
    if status != "OK":
        return "LOW"
    exp = fields["exp"]["date"]
    y, m, d = (int(x) for x in entry["db_exp_date"][:10].split("-"))
    exp_ok = (exp["year"], exp["month"]) == (y, m) and (exp["day"] is None or exp["day"] == d)
    b = fields.get("batch")
    want_b = db_letters(entry["db_no_batch"])
    batch_ok = b is None or want_b is None or b["value"] == want_b or not cf.TOKENS["BATCH3"]
    if b is not None and len(b["value"] or "") != 3:  # batch panjang (26Y070X): tak ada kunci 3 huruf
        batch_ok = True
    return "OK_MATCH_DB" if exp_ok and batch_ok else "OK_DIFF_DB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="holdout")
    args = ap.parse_args()

    out_path = RUNS / "coding_formula" / f"photos_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {e["attachment_id"]: e for e in json.loads(out_path.read_text(encoding="utf-8"))} \
        if out_path.exists() else {}
    cfg = le._code_lines_cfg()
    rows = pick(args.n, args.split, args.seed)
    print(f"{len(rows)} foto, formula dari isian operator; GPU {cfg['url']} {cfg['model']}")
    for i, r in enumerate(rows, 1):
        if r["attachment_id"] in done:
            continue
        img = cv2.imread(str(MASTER / r["image"]))
        try:
            views = read_all_views(img, cfg)
        except (le.LlmUnavailable, le.LlmBadJson) as exc:
            print(f"  [{i}] {r['attachment_id']} ERROR {exc}")
            continue
        e = {k: r[k] for k in ("attachment_id", "batch_id", "field_label", "image", "db_no_batch",
                               "db_exp_date", "typed", "formula")}
        e["views"] = views
        done[e["attachment_id"]] = e
        st, fields, _, _ = simulate(e, SIDES)
        print(f"  [{i}/{len(rows)}] {e['attachment_id']:>5} {verdict(e, st, fields):<11} "
              f"{sum(v['sec'] for v in views):5.1f}s  {e['formula']}  "
              f"{ {f: v.get('value') for f, v in fields.items() if v.get('required')} }")
        out_path.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1), encoding="utf-8")

    entries = [done[r["attachment_id"]] for r in rows if r["attachment_id"] in done]
    print(f"\n=== {len(entries)} foto ===")
    configs = [("4 view 1024,640,1600,2048", SIDES)] + [
        (f"3 view {','.join(map(str, c))}", list(c)) for c in itertools.combinations(SIDES, 3)] + [
        (f"2 view {','.join(map(str, c))}", list(c)) for c in itertools.combinations(SIDES, 2)]
    for name, sides in configs:
        cnt, calls, secs, ok_secs = Counter(), [], [], []
        for e in entries:
            st, fields, n, s = simulate(e, sides)
            v = verdict(e, st, fields)
            cnt[v] += 1
            calls.append(n)
            secs.append(s)
            if st == "OK":
                ok_secs.append(s)
        secs.sort()
        print(f"{name:<28} OK={cnt['OK_MATCH_DB'] + cnt['OK_DIFF_DB']:>3} (cocok DB {cnt['OK_MATCH_DB']}, "
              f"beda DB {cnt['OK_DIFF_DB']})  LOW={cnt['LOW']:>3}  panggilan rata2 {sum(calls) / len(calls):.2f}  "
              f"dtk median {secs[len(secs) // 2]:.1f} rata2 {sum(secs) / len(secs):.1f}"
              + (f"  kasus OK {sum(ok_secs) / len(ok_secs):.1f}" if ok_secs else ""))
    per_side = {s: [v["sec"] for e in entries for v in e["views"] if v["side"] == s] for s in SIDES}
    print("waktu per view (rata2 dtk): " + ", ".join(f"{s}px {sum(t) / len(t):.2f}" for s, t in per_side.items() if t))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
