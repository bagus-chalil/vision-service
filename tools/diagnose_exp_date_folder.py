"""
Batch diagnostic over images/*.{jpg,jpeg,png} for field_type=tube_exp_date
specifically (label_anchor path) - separate from diagnose_images_folder.py
which only exercises tube_emboss_default. Loads PaddleOCR + YOLO directly
(same construction as main.py), runs analyze_label_anchor_field() on every
image, and prints a summary table so regressions across the whole EXP-date
scenario set (compact 6-digit, dotted 4-digit-year, no-label digit fallback,
EXP+MFD adjacent zoom_retry) are visible at once.

This script has no ground truth for most images/ samples - it reports what
the pipeline returns, not whether it's "correct". Read status/extracted
alongside match_method to judge: LABEL_NOT_FOUND on a photo you know has no
EXP text at all is correct behavior, not a failure.

Run: venv\\Scripts\\python.exe tools\\diagnose_exp_date_folder.py
"""

import sys
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import tube_emboss_pipeline as pipeline

IMAGES_DIR = ROOT_DIR / "images"
FIELD_TYPE = "tube_exp_date"


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

    rows = []
    for path in image_paths:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            rows.append((path.name, "READ_FAILED", None, None, None, None))
            continue

        result = pipeline.analyze_tube_emboss(
            image_bgr,
            field_type=FIELD_TYPE,
            request_id=f"diag-exp-{path.stem}",
            ocr_engine=ocr_engine,
            yolo_model=yolo_model,
            debug=False,
            image_ref=path.name,
        )

        status = result.get("status")
        error_reason = result.get("error_reason")
        extracted = result.get("extracted_date_code")
        confidence = result.get("confidence")
        match_method = result.get("match_method")

        print(f"[{path.name}]")
        print(f"  status = {status}, error_reason = {error_reason}, match_method = {match_method}")
        print(f"  extracted_date_code = {extracted!r}, confidence = {confidence}")
        print()

        rows.append((path.name, status or error_reason, extracted, confidence, match_method))

    print("\n=== SUMMARY ===")
    print(f"{'image':45s} {'status':22s} {'extracted':10s} {'conf':6s} {'match_method':15s}")
    for name, status, extracted, confidence, match_method in rows:
        conf_str = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "-"
        print(f"{name:45s} {str(status):22s} {str(extracted):10s} {conf_str:6s} {str(match_method):15s}")


if __name__ == "__main__":
    main()
