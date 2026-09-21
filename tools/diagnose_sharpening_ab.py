"""
A/B comparison: OCR on raw crimp_crop vs OCR on preprocess_crimp_for_ocr()
output, across every image in images/. Does NOT touch main.py or the
production pipeline - this only decides, with data, whether wiring
sharpening into analyze_tube_emboss() is worth doing.

Also reports what fraction of these samples would hit the fast path added
to recognize_crimp_text_dual() (skip the sharpened pass when raw confidence
is already below GEMINI_FALLBACK_THRESHOLD) - same threshold, read straight
from tube_emboss_pipeline so this never drifts out of sync with production.

Run: venv\\Scripts\\python.exe diagnose_sharpening_ab.py
"""

import sys
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

import tube_emboss_pipeline as pipeline

IMAGES_DIR = Path(__file__).parent / "images"
OUT_DIR = Path(__file__).parent / "diagnostic_out_sharpened"


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
            print(f"[{path.name}] FAILED TO READ")
            continue

        localization = pipeline.localize_tube(image_bgr, yolo_model)
        if not localization["detected"]:
            print(f"[{path.name}] NO TUBE DETECTED - skipped")
            continue

        crimp_crop = localization["crimp_crop"]
        sharpened_crop = pipeline.preprocess_crimp_for_ocr(crimp_crop)

        stem = path.stem.replace(" ", "_")
        cv2.imwrite(str(OUT_DIR / f"{stem}_sharpened.png"), sharpened_crop)

        raw_result = pipeline.recognize_crimp_text(crimp_crop, ocr_engine)
        sharp_result = pipeline.recognize_crimp_text(sharpened_crop, ocr_engine)

        raw_text = raw_result.get("text") if raw_result["detected"] else None
        raw_conf = raw_result.get("confidence") if raw_result["detected"] else None
        sharp_text = sharp_result.get("text") if sharp_result["detected"] else None
        sharp_conf = sharp_result.get("confidence") if sharp_result["detected"] else None

        print(f"[{path.name}]")
        print(f"  RAW:      text={raw_text!r:16s} conf={raw_conf}")
        print(f"  SHARPENED: text={sharp_text!r:16s} conf={sharp_conf}")
        print()

        rows.append((path.name, raw_text, raw_conf, sharp_text, sharp_conf))

    print("\n=== SUMMARY ===")
    threshold = pipeline.GEMINI_FALLBACK_THRESHOLD
    header = f"{'image':30s} {'raw_text':14s} {'raw_conf':9s} {'sharp_text':14s} {'sharp_conf':10s} {'delta':7s} {'fast_path':9s}"
    print(header)
    improved = worsened = same_text = fast_path = 0
    for name, raw_text, raw_conf, sharp_text, sharp_conf in rows:
        rc = f"{raw_conf:.3f}" if isinstance(raw_conf, (int, float)) else "-"
        sc = f"{sharp_conf:.3f}" if isinstance(sharp_conf, (int, float)) else "-"
        delta = "-"
        if isinstance(raw_conf, (int, float)) and isinstance(sharp_conf, (int, float)):
            d = sharp_conf - raw_conf
            delta = f"{d:+.3f}"
            if raw_text == sharp_text:
                same_text += 1
            if d > 0.01:
                improved += 1
            elif d < -0.01:
                worsened += 1
        is_fast_path = isinstance(raw_conf, (int, float)) and raw_conf < threshold
        if is_fast_path:
            fast_path += 1
        fp = "yes" if is_fast_path else "no"
        print(f"{name:30s} {str(raw_text):14s} {rc:9s} {str(sharp_text):14s} {sc:10s} {delta:7s} {fp:9s}")

    print(f"\nsame_text={same_text}/{len(rows)}  confidence_improved={improved}  confidence_worsened={worsened}")
    print(f"fast_path (raw_conf < {threshold}, sharpened pass skipped in production)={fast_path}/{len(rows)} ({100 * fast_path / len(rows):.0f}%)")
    print(f"Sharpened crops saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
