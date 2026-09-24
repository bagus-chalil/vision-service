"""
Measures the time/accuracy impact of capping input resolution before OCR on
the label_anchor pipeline (tube_exp_date) - does NOT touch main.py or
tube_emboss_pipeline.py, this only produces data to decide whether adding a
resize cap is worth it.

Why this pipeline specifically: analyze_label_anchor_field() OCRs the whole
tube crop (or whole frame) TWICE (raw pass + sharpened pass, see
recognize_label_anchor_dual() in tube_emboss_pipeline.py) instead of a small
localized ROI like tube_emboss_default's crimp crop - and neither main.py nor
the pipeline downscales the image anywhere before that. If input photos are
native camera resolution, this is the likely dominant cost.

For each image in images/, runs analyze_tube_emboss(field_type="tube_exp_date")
at full resolution (baseline) and again after downscaling to each cap in
RESIZE_CAPS (longest side, aspect ratio preserved, never upscales - same
never-invent-pixels spirit as preprocess_crimp_for_ocr()'s upscale guard).
Reports per-image wall time and extracted result at each cap, then a summary
of average time reduction % and how often the extracted result changed vs
baseline (the accuracy risk side of this tradeoff).

Caveat: not every sample in images/ necessarily has an "EXP" emboss/label in
frame (some are tube_emboss_default MFD samples) - for those this still
measures timing validly, but the "does the extracted text change" accuracy
signal is only meaningful for genuine EXP samples. Check field_type/known
ground truth per image manually before trusting the accuracy column too far.

Run: venv\\Scripts\\python.exe tools\\diagnose_resize_cap.py
"""

import sys
import time
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import tube_emboss_pipeline as pipeline

IMAGES_DIR = ROOT_DIR / "images"
FIELD_TYPE = "tube_exp_date"

# None = original resolution (baseline). Others = max(width, height) cap in
# pixels, downscaled with aspect ratio preserved, INTER_AREA (correct choice
# for shrinking, unlike the pipeline's own INTER_CUBIC which is for the
# opposite direction - upscaling small crimp crops).
RESIZE_CAPS = [None, 2000, 1600, 1200, 800]


def resize_to_cap(image_bgr, cap):
    if cap is None:
        return image_bgr
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= cap:
        return image_bgr
    scale = cap / longest
    return cv2.resize(
        image_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )


def main():
    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Loading models...")
    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )
    yolo_model = pipeline.load_yolo_model()
    print("Models ready.\n")

    all_rows = []  # (image_name, cap_label, elapsed_s, status, extracted, confidence, match_method)
    for path in image_paths:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            print(f"[{path.name}] FAILED TO READ")
            continue

        h, w = image_bgr.shape[:2]
        print(f"[{path.name}] native size {w}x{h}")

        baseline_extracted = None
        for cap in RESIZE_CAPS:
            resized = resize_to_cap(image_bgr, cap)
            cap_label = "original" if cap is None else f"cap{cap}"

            start = time.perf_counter()
            result = pipeline.analyze_tube_emboss(
                resized,
                field_type=FIELD_TYPE,
                request_id=f"diag-{path.stem}-{cap_label}",
                ocr_engine=ocr_engine,
                yolo_model=yolo_model,
                debug=False,
                image_ref=path.name,
            )
            elapsed = time.perf_counter() - start

            extracted = result.get("extracted_date_code") or result.get("raw_ocr_text")
            status = result.get("status") or result.get("error_reason")
            confidence = result.get("confidence")
            match_method = result.get("match_method")

            if cap is None:
                baseline_extracted = extracted
            changed = "yes" if (cap is not None and extracted != baseline_extracted) else ""

            rh, rw = resized.shape[:2]
            print(
                f"  {cap_label:10s} ({rw}x{rh}) time={elapsed:6.2f}s "
                f"status={str(status):16s} extracted={str(extracted):10s} "
                f"conf={confidence} method={match_method} {('<-- CHANGED' if changed else '')}"
            )
            all_rows.append((path.name, cap_label, elapsed, status, extracted, confidence, match_method))
        print()

    print("\n=== SUMMARY (avg time per cap, vs baseline) ===")
    baseline_times = {name: e for name, cap_label, e, *_ in all_rows if cap_label == "original"}
    for cap in RESIZE_CAPS:
        cap_label = "original" if cap is None else f"cap{cap}"
        times = [e for name, cl, e, *_ in all_rows if cl == cap_label]
        if not times:
            continue
        avg_time = sum(times) / len(times)
        if cap is None:
            print(f"{cap_label:10s} avg_time={avg_time:6.2f}s (baseline)")
        else:
            reductions = [
                100 * (1 - e / baseline_times[name])
                for name, cl, e, *_ in all_rows
                if cl == cap_label and name in baseline_times and baseline_times[name] > 0
            ]
            avg_reduction = sum(reductions) / len(reductions) if reductions else 0
            changed_count = sum(
                1 for name, cl, e, status, extracted, conf, mm in all_rows
                if cl == cap_label and extracted != next(
                    ex for n2, cl2, e2, s2, ex, c2, m2 in all_rows if n2 == name and cl2 == "original"
                )
            )
            print(
                f"{cap_label:10s} avg_time={avg_time:6.2f}s "
                f"avg_time_reduction={avg_reduction:5.1f}% "
                f"extracted_text_changed_vs_baseline={changed_count}/{len(times)}"
            )


if __name__ == "__main__":
    main()
