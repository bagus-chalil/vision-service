"""Lokasi data uji privat (satu sumber). Dev-only, dipakai skrip di tools/.

testdata_private/            (GITIGNORED - foto & data produksi, jangan di-commit)
  master/  images/<use_case>/<stage>__<field_label>/<attachment_id>.jpg + manifest.jsonl
           -> SATU-SATUNYA tempat foto; dibangun tools/build_master_dataset.py
  keys/    kunci jawaban & set eval (JSON); field `path` relatif ke root repo,
           menunjuk ke master/images (atau fixture repo images/)
  runs/    hasil eksperimen/eval & log (arsip; skrip menulis output baru ke sini)
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TDP = ROOT / "testdata_private"
MASTER = TDP / "master"
KEYS = TDP / "keys"
RUNS = TDP / "runs"


@lru_cache(maxsize=1)
def _manifest():
    out = {}
    with open(MASTER / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["attachment_id"]] = r
    return out


def manifest_row(attachment_id):
    return _manifest().get(int(attachment_id))


def image_path(attachment_id):
    """Path absolut foto layak di master, atau None kalau dibuang/tidak ada."""
    r = manifest_row(attachment_id)
    return str(MASTER / r["image"]) if r and r.get("image") else None


def resolve(path):
    """Path di file kunci (relatif ke root repo atau absolut) -> path absolut (str)."""
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def load_key(name):
    return json.loads((KEYS / name).read_text(encoding="utf-8"))
