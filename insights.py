"""Text insights for consumer apps (ipc_app, new_trial_validation_app).

The Vision Service is the single AI provider: a consumer sends already-prepared, anonymised FACTS
(plain sentences it computed itself) plus a `kind`, and gets a short written paragraph back. The
prompt, the writing rules and the output check all live here, so every consumer gets the same
behaviour and a prompt fix is one deploy in one place. Consumers never send a prompt of their own.

Rules baked into the kinds:
  * every kind: use only the facts that were sent, add nothing;
  * kinds about an approval decision (`guard_decisions: True`, with ADVISORY_RULES in their system
    prompt): never recommend approve / reject / hold. `violates_advisory` rejects an answer that does,
    so the consumer falls back to its own rule-based text. Kinds that are not about a decision (e.g.
    `ipc_client_issues`, an admin reading an error log) set `guard_decisions: False`, because there
    words like "reject" or "hold" are ordinary vocabulary and must not be refused.

Adding a kind = one entry in KINDS. Nothing is stored here.
"""
import re

import llm_extract

MAX_FACTS = 40
MAX_FACT_CHARS = 600
MAX_TOTAL_CHARS = 8000

# Language the paragraph is written in. The facts may already be in that language; the consumer
# picks it (default id keeps older consumers working unchanged).
LANGUAGES = {"id": "Bahasa Indonesia", "en": "English", "ko": "Korean (formal business style)"}

ADVISORY_RULES = (
    "Aturan: pakai HANYA fakta pada daftar yang diberikan; jangan menambah angka, penyebab, atau "
    "kesimpulan yang tidak ada di sana. Jangan menyarankan approve, reject, hold, atau keputusan apa "
    "pun: keputusan sepenuhnya di tangan manusia. Jika tidak ada temuan, katakan itu dengan singkat."
)

KINDS = {
    "ipc_batch_summary": {
        "label": "Ringkasan batch untuk approver IPC",
        "system": (
            "Kamu membantu approver QC di pabrik kosmetik membaca ringkasan hasil pemeriksaan satu batch. "
            "Tulis dalam Bahasa Indonesia, satu paragraf singkat (maksimal sekitar 120 kata), nada netral dan jelas. "
            "Soroti dulu temuan NOT OK, terutama yang ditandai 'Berulang', lalu hal lain yang perlu dicek approver. "
            + ADVISORY_RULES
        ),
        "intro": "Hasil pemeriksaan batch:",
        "guard_decisions": True,
    },
    "trial_summary": {
        "label": "Ringkasan trial untuk approver dan reviewer",
        "system": (
            "Kamu membantu approver dan reviewer departemen di pabrik kosmetik membaca ringkasan satu trial validasi. "
            "Tulis satu paragraf singkat (maksimal sekitar 120 kata), nada netral dan jelas. "
            "Soroti dulu hasil validasi yang NOT OK dan catatan reviewer yang bukan sekadar 'OK', lalu hal lain yang perlu dicek. "
            "Hasil weighing yang tidak dilakukan (skip) dan parameter N/A adalah hal biasa, bukan temuan. "
            + ADVISORY_RULES
        ),
        "intro": "Data trial:",
        "guard_decisions": True,
    },
    "ipc_client_issues": {
        "label": "Analisis log kendala lapangan IPC",
        "system": (
            "Kamu membantu admin aplikasi pemeriksaan kualitas pabrik kosmetik (IPC) membaca log kendala dari tablet/HP operator. "
            "Jawab dalam Bahasa Indonesia, singkat dan konkret (maksimal sekitar 250 kata), dengan format:\n"
            "1. Ringkasan: satu-dua kalimat tentang pola utama.\n"
            "2. Per kendala penting (urut dari paling mendesak): dugaan penyebab dan satu langkah cek yang bisa dilakukan admin.\n"
            "Aturan: gunakan hanya data yang diberikan, jangan mengarang penyebab yang tidak didukung data. "
            "Jika data belum cukup, katakan apa yang perlu dikumpulkan.\n"
            "Petunjuk: banyak kejadian dari satu perangkat/browser saja biasanya masalah perangkat atau browser lama. "
            "Kejadian dengan status offline atau koneksi 2g/slow-2g biasanya jaringan. "
            "Kejadian di semua perangkat biasanya bug aplikasi atau server."
        ),
        "intro": "Data kendala terbuka:",
        "guard_decisions": False,
    },
}

# A recommendation verb followed (within a few words) by a decision word, e.g. "sebaiknya di-approve",
# "disarankan reject", "layak disetujui". Restating a recorded decision ("Keputusan QC: Packing: Hold")
# does not match, because no recommendation word precedes it.
_RECOMMEND = re.compile(
    r"\b(sebaiknya|disarankan|saya\s+sarankan|menyarankan|direkomendasikan|rekomendasi(?:nya)?|layak|dapat|bisa)\b"
    r"\W+(?:\w+\W+){0,3}?"
    r"(approve|di-?approve|approved|reject|di-?reject|rejected|hold|di-?hold|disetujui|setujui|menyetujui|ditolak|tolak|menolak)",
    re.IGNORECASE,
)

# English / Korean equivalents (the facts and answer may be written in those languages).
_RECOMMEND_EN = re.compile(
    r"\b(recommend\w*|should|suggest\w*|advis\w*|better\s+to|ought\s+to)\b"
    r"\W+(?:\w+\W+){0,3}?"
    r"(approv\w*|reject\w*|hold|declin\w*)",
    re.IGNORECASE,
)
_RECOMMEND_KO = re.compile(
    r"(승인|반려|보류|거절)\S*\s*(?:\S+\s+){0,2}?(권장|권고|추천|바람직|하는\s*것이\s*좋|해야)"
    r"|(권장|권고|추천)\S*\s*(?:\S+\s+){0,2}?(승인|반려|보류|거절)",
)


class InsightError(ValueError):
    """The request is not acceptable (unknown kind, empty or oversized facts)."""


def kinds():
    return [{"kind": key, "label": cfg["label"]} for key, cfg in KINDS.items()]


def clean_facts(facts):
    """Strip, drop blanks and enforce the size limits. Raises InsightError."""
    cleaned = [re.sub(r"\s+", " ", str(f)).strip() for f in (facts or [])]
    cleaned = [f for f in cleaned if f]
    if not cleaned:
        raise InsightError("facts kosong")
    if len(cleaned) > MAX_FACTS:
        raise InsightError(f"terlalu banyak fakta (maks {MAX_FACTS})")
    if any(len(f) > MAX_FACT_CHARS for f in cleaned):
        raise InsightError(f"satu fakta terlalu panjang (maks {MAX_FACT_CHARS} karakter)")
    if sum(len(f) for f in cleaned) > MAX_TOTAL_CHARS:
        raise InsightError("total fakta terlalu panjang")
    return cleaned


def build_prompt(kind, facts, language="id"):
    cfg = KINDS.get(kind)
    if cfg is None:
        raise InsightError(f"kind tidak dikenal: {kind}")
    if language not in LANGUAGES:
        raise InsightError(f"language tidak dikenal: {language}")
    body = "\n".join(f"- {f}" for f in clean_facts(facts))
    system = cfg["system"]
    if language != "id":
        system += f" Tulis jawaban dalam {LANGUAGES[language]}."
    return system, f"{cfg['intro']}\n{body}"


def violates_advisory(text):
    text = text or ""
    return bool(_RECOMMEND.search(text) or _RECOMMEND_EN.search(text) or _RECOMMEND_KO.search(text))


def summarize(kind, facts, language="id"):
    """Returns the written paragraph. Raises InsightError (bad request), llm_extract.LlmUnavailable
    (Ollama down, propagates) or AdvisoryViolation (the model recommended a decision)."""
    system, prompt = build_prompt(kind, facts, language)
    text = llm_extract.chat_text(system, prompt)
    if not text.strip():
        raise llm_extract.LlmUnavailable("jawaban kosong")
    if KINDS[kind].get("guard_decisions", True) and violates_advisory(text):
        raise AdvisoryViolation(text)
    return text.strip()


class AdvisoryViolation(Exception):
    """The answer recommended approve/reject/hold, which the Vision Service never lets through."""
