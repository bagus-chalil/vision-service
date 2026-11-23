"""Unit test voting Nomor IM di _build_im_result (tanpa GPU/Ollama). Jalankan: python tests/test_im_vote.py

Data view diambil dari bacaan asli ipc_app (ipc_logs batch 911 / 913, 2026-11-23)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_extract as le  # noqa: E402

SHELF = [{"years": 3, "months": None, "raw_text": "(3 Tahun dari Tanggal Mixing)", "packaging": "Bottle"}]


def view(im, legible=True):
    return {"im_number": im, "shelf_life": SHELF, "legible": {"im_number": legible, "shelf_life": True}}


def build(ims):
    return le._build_im_result([view(i) if not isinstance(i, tuple) else view(*i) for i in ims], len(ims), [{"model": "t"}], 1.0)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


TONER = "IM-KKI NUR NPURE LICORICE BRIGHT & GLOW TONER 80ML"

# batch 911 (log 58): 3 dari 4 sepakat -> dipercaya, status OK
r = build([TONER, "IM-KKI NUP PURE LICORICE BRIGHT & GLOW TONER 80ML", TONER, TONER])
check(r["view_agreement"]["im_number"] is True, "3/4 sepakat harus dipercaya")
check(r["im_number"] == TONER, r["im_number"])
check(r["status"] == "OK", r["status"])
check(r["im_number_votes"] == {"agree": 3, "views": 4}, r["im_number_votes"])

# view pertama yang kalah suara: nilai diambil dari pemenang, bukan view pertama
r = build(["IM-KKI NUP PURE LICORICE BRIGHT & GLOW TONER 80ML", TONER, TONER, TONER])
check(r["im_number"] == TONER and r["view_agreement"]["im_number"], r["im_number"])

# batch 911 (log 55): hanya 2 dari 4 sepakat -> tidak dipercaya
r = build([TONER, "IM-KKI NUR PURE LICORICE BRIGHT & GLOW TONER 80ML", "IM-KKI NUPURE LICORICE BRIGHT & GLOW TONER 80ML", TONER])
check(r["view_agreement"]["im_number"] is False, "2/4 tidak boleh dipercaya")
check(r["im_number"] == TONER, "nilai tetap dilaporkan (view pertama)")
check(r["status"] == "LOW_CONFIDENCE", r["status"])

# beda spasi / tanda '-' saja bukan ketidaksepakatan
r = build(["IM-NUR-NPURE PAD 130G - C", "IM NUR NPURE PAD 130G-C", "IM-NUR-NPURE PAD 130G - C", "IM-NUR-NPURE PAD 130G - C"])
check(r["view_agreement"]["im_number"] is True, "beda spasi harus sepakat")

# view yang tidak legible tidak ikut memilih; 2 legible + 2 tidak -> tidak dipercaya
r = build([TONER, (TONER, False), (TONER, False), TONER])
check(r["view_agreement"]["im_number"] is False, "hanya 2 suara legible")

# semua null -> tidak dipercaya, tidak error
r = build([None, None, None, None])
check(r["im_number"] is None and r["view_agreement"]["im_number"] is False, "null")

print("OK test_im_vote")
