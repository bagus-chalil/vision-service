"""
Batch diagnostic over images/*.jpg using the real tube_emboss_pipeline
(no server needed - loads PaddleOCR + YOLO directly, same construction as
main.py). For each image, saves:
  debug_output/diagnostic_out/<name>_tube.png   - YOLO tube bbox crop
  debug_output/diagnostic_out/<name>_crimp.png  - top-15% crimp crop (what OCR actually sees)
and prints raw_ocr_text / confidence / validation per image, plus a summary
table at the end so ridge-margin patterns across all photos are visible at
once (not just the 2 old samples).

Run: venv\\Scripts\\python.exe tools\\diagnose_images_folder.py
"""

import sys
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import tube_emboss_pipeline as pipeline

IMAGES_DIR = ROOT_DIR / "images"
OUT_DIR = ROOT_DIR / "debug_output" / "diagnostic_out"


def main():
    OUT_DIR.mkdir(exist_ok=True)

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
            print(f"[{path.name}] FAILED TO READ (check path/encoding)")
            rows.append((path.name, "READ_FAILED", None, None, None))
            continue

        result = pipeline.analyze_tube_emboss(
            image_bgr,
            field_type=pipeline.DEFAULT_FIELD_TYPE,
            request_id=f"diag-{path.stem}",
            ocr_engine=ocr_engine,
            yolo_model=yolo_model,
            debug=True,
            image_ref=path.name,
        )

        debug = result.get("debug", {})
        stem = path.stem.replace(" ", "_")

        # Re-run localize_tube directly to get crop arrays for saving
        # (analyze_tube_emboss only returns base64 in debug, decode is
        # wasteful - just recompute the cheap crop step).
        localization = pipeline.localize_tube(image_bgr, yolo_model)
        if localization["detected"]:
            cv2.imwrite(str(OUT_DIR / f"{stem}_tube.png"), localization["tube_crop"])
            cv2.imwrite(str(OUT_DIR / f"{stem}_crimp.png"), localization["crimp_crop"])
            tube_h, tube_w = localization["tube_crop"].shape[:2]
            crimp_h = localization["crimp_crop"].shape[0]
        else:
            tube_h = tube_w = crimp_h = None

        status = result.get("status")
        raw_text = result.get("raw_ocr_text")
        confidence = result.get("confidence")
        error_reason = result.get("error_reason")
        tube_conf = result.get("tube_detection_confidence") or (localization.get("confidence"))

        print(f"[{path.name}]")
        print(f"  tube_detection_confidence = {tube_conf}")
        print(f"  tube_crop = {tube_w}x{tube_h}, crimp_crop_height = {crimp_h}")
        print(f"  status = {status}, error_reason = {error_reason}")
        print(f"  raw_ocr_text = {raw_text!r}, confidence = {confidence}")
        if result.get("validation"):
            print(f"  validation = {result['validation']}")
        print()

        rows.append((path.name, status or error_reason, raw_text, confidence, tube_conf))

    print("\n=== SUMMARY ===")
    print(f"{'image':30s} {'status':16s} {'raw_text':14s} {'conf':6s} {'tube_conf':9s}")
    for name, status, raw_text, confidence, tube_conf in rows:
        conf_str = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "-"
        tube_conf_str = f"{tube_conf:.3f}" if isinstance(tube_conf, (int, float)) else "-"
        print(f"{name:30s} {str(status):16s} {str(raw_text):14s} {conf_str:6s} {tube_conf_str:9s}")

    print(f"\nDebug crops saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
