"""Sync + kelompokkan + bersihkan foto produksi ipc_app jadi satu dataset induk.

Sumber : Downloads/ipc-attachments (salinan storage prod) + DB lokal ipc_system
         (SELECT saja; sinkronkan DB dari prod DULU, jenis foto hanya dari DB).
Output : testdata_private/master/   (GITIGNORED - foto & data produksi, jangan di-commit)
  images/<use_case>/<stage>__<field_label>/<attachment_id>.jpg   hanya foto LAYAK
  manifest.jsonl   satu baris per ipc_attachments (layak + dibuang + file hilang)
  manifest.csv     sama, untuk dibuka di Excel
  summary.json     hitungan per use_case / alasan buang / split

Idempoten: jalankan ulang setelah download foto baru + sync DB. File yang sudah ada
(ukuran sama) tidak disalin ulang; file di images/ yang tidak lagi layak dihapus.
Struktur testdata_private (master/ keys/ runs/): lihat tools/dataset.py.
PERINGATAN: kalau foto yang dirujuk keys/*.json berubah jadi tidak layak, path di
key itu jadi tidak ada - cek `missing_key_images` di output.

Aturan buang (status=excluded, foto TIDAK disalin):
  test_data       produk "Sample Product*" atau batch terhapus dgn no_batch dummy
  missing_file    baris DB ada, file tidak ada di download
  unreadable      gagal di-decode
  blank_black     frame hitam (mean<10 & std<5) - kamera belum siap
  blank_uniform   >=90% piksel satu warna (frame polos / placeholder "kamera tidak tersedia")
  blank_dark      mean<20 & std<17 - terlalu gelap, tidak ada isi terbaca
  featureless     std<6 atau var Laplacian<3 atau edge_frac<0.001 (Canny, 256px) - frame
                  polos/blur total tanpa struktur (juga utk color: swatch asli selalu punya tepi)
  (ambang di atas dicek visual 2026-11-19 lewat contact sheet; frame di batas ambang
   semuanya sampah, dan edge_frac punya celah tegas: 150 foto <0.001, cuma +2 s/d 0.004.
   Belum ada kunci manusia - kalau ragu, lihat manifest.csv)
  duplicate       sha1 identik dgn foto lain di field_label sama (yang id terkecil dipertahankan)
Flag (status=usable, foto tetap disalin, cek sebelum dipakai sbg kunci):
  batch_deleted        batch terhapus tapi data asli -> db_key_trusted=false
  no_db_key            tidak ada kunci DB utk use_case ini
  nonstandard_no_batch no_batch tidak berpola NNXnnn[/XXX] (mis. "ABB", "AKB 26X001")
  very_dark            mean<35
  very_blurry          variance Laplacian < 15 (ambang kasar, belum dikalibrasi)
  low_detail           edge_frac < 0.012 - campuran sampah (jari/kilau) & foto asli objek
                       kecil; cek mata sebelum dipakai
  cross_field_duplicate foto identik juga diunggah di field_label lain (salah slot?)
  near_duplicate       dhash mirip (<=4 bit) foto lain di batch+field sama (jepretan berulang)
Split (hanya foto layak): per BATCH (bukan per foto, cegah bocor), sha1(batch_id) % 100 < 20
  -> "eval" (hold-out), selain itu "train". `used_in_keys` = nama file keys/*.json yang
  memuat foto ini (sudah pernah dipakai tuning/eval).
"""
import csv, hashlib, json, re, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"C:\Users\cosmax\Downloads")  # file_path DB = "ipc-attachments/..."
TDP = ROOT / "testdata_private"
OUT = TDP / "master"
KEYS = TDP / "keys"
IMG = OUT / "images"
MYSQL = next(Path(r"C:\laragon\bin\mysql").glob("*/bin/mysql.exe"))
EVAL_PCT = 20

USE_CASE = {
    "date_bulk": "label_bulk", "im_number": "im_document", "image_tube": "tube",
    "coding": "coding", "coding_primary": "coding", "coding_secondary": "coding",
    "coding_tersier": "coding", "primary_coding_batch_exp": "coding",
    "secondary_coding_batch_exp": "coding", "tersier_coding_batch": "coding",
    "wo_image": "wo_number", "wi_number": "wi_number", "exp_date": "finished_exp_date",
    "color": "color", "temperature_setting": "other", "palletisasi": "other",
}
REAL_BATCH = re.compile(r"\d{2}[A-Z]{1,2}\d")
STD_BATCH = re.compile(r"^\d{2}[A-Z]{1,2}\d{3}(/[A-Z]{3})?$")


def q(sql):
    r = subprocess.run([str(MYSQL), "-uroot", "ipc_system", "-B", "-N", "-e", sql],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        sys.exit(r.stderr)
    return [[None if v == "NULL" else v for v in l.split("\t")] for l in r.stdout.splitlines()]


def db_key(uc, r):
    """Kunci DB yang relevan per use_case (None = tidak ada kunci)."""
    if uc == "label_bulk" and r["db_no_batch"] and r["db_mixing_date"]:
        return {"no_batch": r["db_no_batch"], "mixing_date": r["db_mixing_date"], "bulk_code": r["db_bulk_code"]}
    if uc == "im_document" and r["db_shelf_life_months"]:
        return {"shelf_life_months": r["db_shelf_life_months"]}
    if uc in ("coding", "tube", "finished_exp_date") and r["db_exp_date"] and r["db_no_batch"]:
        return {"no_batch": r["db_no_batch"], "exp_date": r["db_exp_date"]}
    return None


def metrics(path):
    data = np.fromfile(str(path), dtype=np.uint8)  # aman utk path Windows (gotcha #4)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (256, int(256 * h / w)) if w >= h else (int(256 * w / h), 256), interpolation=cv2.INTER_AREA)
    med = float(np.median(s))
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(int)
    dh = sum(1 << i for i, v in enumerate((small[:, 1:] > small[:, :-1]).flatten()) if v)
    return dict(width=w, height=h, mean=round(float(s.mean()), 1), std=round(float(s.std()), 1),
                dominant_frac=round(float((np.abs(s.astype(int) - med) <= 6).mean()), 3),
                lap_var=round(float(cv2.Laplacian(s, cv2.CV_64F).var()), 1),
                edge_frac=round(float((cv2.Canny(s, 50, 150) > 0).mean()), 4), dhash=dh)


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main():
    rows = q(
        "select a.id, a.ipc_batch_id, a.stage, a.field_label, a.file_path, a.created_at, "
        "b.no_batch, b.mixing_date, b.exp_date, coalesce(b.shelf_life_months, p.shelf_life_months), "
        "b.bulk_code, b.flow_type, b.primary_packaging_type, b.deleted_at is not null, "
        "p.fg_code, p.product_name, sc.coding_format_primary, sc.coding_format_secondary, sc.coding_format_tersier "
        "from ipc_attachments a left join ipc_batches b on b.id=a.ipc_batch_id "
        "left join master_products p on p.id=b.master_product_id "
        "left join startup_checks sc on sc.ipc_batch_id=b.id order by a.id")
    cols = ["attachment_id", "batch_id", "stage", "field_label", "file_path", "uploaded_at",
            "db_no_batch", "db_mixing_date", "db_exp_date", "db_shelf_life_months", "db_bulk_code",
            "flow_type", "db_packaging_type", "batch_deleted", "fg_code", "product",
            "coding_format_primary", "coding_format_secondary", "coding_format_tersier"]
    feedback = defaultdict(list)
    for fb in q("select ipc_batch_id, field_type, field_key, ai_value, ai_trusted, final_value, source, outcome "
                "from ipc_ai_field_feedback order by id"):
        feedback[fb[0]].append(dict(zip(["field_type", "field_key", "ai_value", "ai_trusted", "final_value",
                                         "source", "outcome"], fb[1:])))

    # foto mana yang sudah pernah dipakai tuning/eval: attachment id di testdata_private/keys/*.json
    used = defaultdict(list)
    for f in sorted(KEYS.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for it in (d.get("items", []) if isinstance(d, dict) else d):
            aid = it.get("attachment_id", it.get("id")) if isinstance(it, dict) else None
            if isinstance(aid, int):
                used[aid].append(f.stem)

    # klasifikasi Qwen tube (keys/tube_classify*.json), per attachment id
    tube_cls = {}
    for name in ("tube_classify.json", "tube_classify_v2.json"):
        f = KEYS / name
        if f.exists():
            for e in json.loads(f.read_text(encoding="utf-8")):
                if "error" not in e and "container" in e:
                    tube_cls[int(e["id"])] = {k: e.get(k) for k in ("container", "crimp_visible", "code_on_crimp",
                                                                     "black_mark_over_code", "printed_code_on_body")}

    items = []
    for raw in rows:
        r = dict(zip(cols, raw))
        r["attachment_id"] = int(r["attachment_id"])
        r["batch_id"] = int(r["batch_id"]) if r["batch_id"] else None
        r["batch_deleted"] = r["batch_deleted"] == "1"
        uc = r["use_case"] = USE_CASE.get(r["field_label"], "other")
        r["flags"], r["exclude_reason"] = [], None
        nb = r["db_no_batch"] or ""
        if (r["product"] or "").lower().startswith("sample product") or (r["batch_deleted"] and not REAL_BATCH.search(nb)):
            r["exclude_reason"] = "test_data"
        src = SRC / r["file_path"]
        if not r["exclude_reason"] and not src.exists():
            r["exclude_reason"] = "missing_file"
        if not r["exclude_reason"]:
            m = metrics(src)
            if m is None:
                r["exclude_reason"] = "unreadable"
            else:
                r.update(m)
                r["sha1"] = sha1(src)
                r["file_bytes"] = src.stat().st_size
                if m["mean"] < 10 and m["std"] < 5:
                    r["exclude_reason"] = "blank_black"
                elif m["dominant_frac"] >= 0.9:
                    r["exclude_reason"] = "blank_uniform"
                elif m["mean"] < 20 and m["std"] < 17:
                    r["exclude_reason"] = "blank_dark"
                elif m["std"] < 6 or m["lap_var"] < 3 or m["edge_frac"] < 0.001:
                    r["exclude_reason"] = "featureless"
        if r["batch_deleted"]:
            r["flags"].append("batch_deleted")
        key = None if r["batch_deleted"] else db_key(uc, r)
        r["db_key"] = key
        r["db_key_trusted"] = key is not None
        if key is None and not r["batch_deleted"]:
            r["flags"].append("no_db_key")
        if nb and not STD_BATCH.match(nb.replace("  ", " ").strip()):
            r["flags"].append("nonstandard_no_batch")
        if r.get("mean") is not None and not r["exclude_reason"]:
            if r["mean"] < 35:
                r["flags"].append("very_dark")
            if r["lap_var"] < 15:
                r["flags"].append("very_blurry")
            if r["edge_frac"] < 0.012:
                r["flags"].append("low_detail")
        if r["batch_id"] in feedback:
            r["ai_feedback"] = feedback[r["batch_id"]]
        if r["attachment_id"] in tube_cls:
            r["qwen_tube_classify"] = tube_cls[r["attachment_id"]]
        items.append(r)

    # duplikat persis
    first = {}
    by_sha = defaultdict(set)
    for r in items:
        if r["exclude_reason"] or "sha1" not in r:
            continue
        by_sha[r["sha1"]].add(r["field_label"])
        k = (r["sha1"], r["field_label"])
        if k in first:
            r["exclude_reason"], r["duplicate_of"] = "duplicate", first[k]
        else:
            first[k] = r["attachment_id"]
    # dhash mirip di batch+field sama
    groups = defaultdict(list)
    for r in items:
        if not r["exclude_reason"] and "sha1" in r:
            if len(by_sha[r["sha1"]]) > 1:
                r["flags"].append("cross_field_duplicate")
            groups[(r["batch_id"], r["field_label"])].append(r)
    for g in groups.values():
        for i, a in enumerate(g):
            for b in g[:i]:
                if bin(a["dhash"] ^ b["dhash"]).count("1") <= 4:
                    a["flags"].append("near_duplicate")
                    a["near_duplicate_of"] = b["attachment_id"]
                    break

    # split + salin
    keep = set()
    for r in items:
        r["status"] = "excluded" if r["exclude_reason"] else "usable"
        if r["status"] != "usable":
            continue
        r["split"] = "eval" if int(hashlib.sha1(str(r["batch_id"]).encode()).hexdigest()[:8], 16) % 100 < EVAL_PCT else "train"
        r["used_in_keys"] = sorted(set(used.get(r["attachment_id"], [])))
        rel = Path(r["use_case"]) / f"{r['stage']}__{r['field_label']}" / f"{r['attachment_id']}.jpg"
        r["image"] = ("images" / rel).as_posix()
        dst = IMG / rel
        keep.add(dst)
        if not (dst.exists() and dst.stat().st_size == r["file_bytes"]):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / r["file_path"], dst)
    removed = 0
    if IMG.exists():
        for p in IMG.rglob("*"):
            if p.is_file() and p not in keep:
                p.unlink()
                removed += 1
        for p in sorted(IMG.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()

    for r in items:
        r.pop("dhash", None)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    csv_cols = ["attachment_id", "batch_id", "use_case", "stage", "field_label", "status", "exclude_reason",
                "split", "flags", "db_key_trusted", "db_no_batch", "db_mixing_date", "db_exp_date",
                "db_shelf_life_months", "product", "width", "height", "mean", "lap_var", "edge_frac",
                "used_in_keys", "image", "file_path"]
    with open(OUT / "manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for r in items:
            w.writerow(["|".join(r.get(c) or []) if c in ("flags", "used_in_keys") else r.get(c) for c in csv_cols])

    usable_ids = {r["attachment_id"] for r in items if r["status"] == "usable"}
    missing_keys = sorted({f"{k}:{aid}" for aid, ks in used.items() if aid not in usable_ids for k in ks})
    summ = {"total_db_rows": len(items), "excluded_by_reason": Counter(r["exclude_reason"] for r in items if r["exclude_reason"]),
            "removed_stale_files": removed, "missing_key_images": missing_keys, "per_use_case": {}}
    for uc in sorted({r["use_case"] for r in items}):
        rs = [r for r in items if r["use_case"] == uc]
        us = [r for r in rs if r["status"] == "usable"]
        summ["per_use_case"][uc] = {
            "total": len(rs), "usable": len(us),
            "excluded": Counter(r["exclude_reason"] for r in rs if r["exclude_reason"]),
            "usable_with_trusted_db_key": sum(r["db_key_trusted"] for r in us),
            "split": Counter(r["split"] for r in us),
            "eval_with_key_not_used_before": sum(r["split"] == "eval" and r["db_key_trusted"] and not r["used_in_keys"] for r in us),
            "flags": Counter(f for r in us for f in r["flags"]),
            "by_field_label": Counter(f"{r['stage']}__{r['field_label']}" for r in us),
        }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
