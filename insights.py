"""Text insights for consumer apps (ipc_app, new_trial_validation_app).

The Vision Service is the single AI provider: a consumer sends already-prepared, anonymised FACTS
(plain sentences it computed itself) plus a `kind`, and gets a short written paragraph back. The
prompt, the writing rules and the output check all live here, so every consumer gets the same
behaviour and a prompt fix is one deploy in one place. Consumers never send a prompt of their own.

Hard rules baked into every kind (see ADVISORY_RULES):
  * use only the facts that were sent, add nothing;
  * never recommend a decision (approve / reject / hold). The check in `violates_advisory` rejects
    an answer that does, so the consumer falls back to its own rule-based text.

Adding a kind = one entry in KINDS. Nothing is stored here.
"""
import re

import llm_extract

MAX_FACTS = 40
MAX_FACT_CHARS = 600
MAX_TOTAL_CHARS = 8000

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


def build_prompt(kind, facts):
    cfg = KINDS.get(kind)
    if cfg is None:
        raise InsightError(f"kind tidak dikenal: {kind}")
    body = "\n".join(f"- {f}" for f in clean_facts(facts))
    return cfg["system"], f"{cfg['intro']}\n{body}"


def violates_advisory(text):
    return bool(_RECOMMEND.search(text or ""))


def summarize(kind, facts):
    """Returns the written paragraph. Raises InsightError (bad request), llm_extract.LlmUnavailable
    (Ollama down, propagates) or AdvisoryViolation (the model recommended a decision)."""
    system, prompt = build_prompt(kind, facts)
    text = llm_extract.chat_text(system, prompt)
    if not text.strip():
        raise llm_extract.LlmUnavailable("jawaban kosong")
    if violates_advisory(text):
        raise AdvisoryViolation(text)
    return text.strip()


class AdvisoryViolation(Exception):
    """The answer recommended approve/reject/hold, which the Vision Service never lets through."""
