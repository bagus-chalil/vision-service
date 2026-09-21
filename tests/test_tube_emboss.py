"""
Minimal manual test for the tube emboss sub-pipeline endpoint.

Start the server first:
    venv\\Scripts\\uvicorn.exe main:app --reload --port 8000

Then run:
    venv\\Scripts\\python.exe tests\\test_tube_emboss.py C:\\path\\to\\tube_photo.jpg [field_type]

Prints the JSON response. If debug crops are present in the response, saves
them to debug_output/ at the project root as tube_crop_debug.png /
crimp_crop_debug.png so you can eyeball whether the YOLO crop and the 15%
top-crop landed correctly.
"""

import base64
import json
import sys
import uuid
from pathlib import Path

import requests

SERVER_URL = "http://127.0.0.1:8000/api/tube-emboss/analyze"


def save_debug_crop(data_uri: str, out_path: Path):
    if not data_uri or "," not in data_uri:
        return
    _, b64_data = data_uri.split(",", 1)
    out_path.write_bytes(base64.b64decode(b64_data))
    print(f"Saved debug crop -> {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_tube_emboss.py <image_path> [field_type]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    field_type = sys.argv[2] if len(sys.argv) > 2 else "tube_emboss_default"

    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        data = {
            "request_id": str(uuid.uuid4()),
            "field_type": field_type,
            "debug": "true",
        }
        response = requests.post(SERVER_URL, files=files, data=data, timeout=60)

    response.raise_for_status()
    result = response.json()

    debug = result.pop("debug", None)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if debug:
        out_dir = Path(__file__).parent.parent / "debug_output"
        out_dir.mkdir(exist_ok=True)
        save_debug_crop(debug.get("tube_crop_base64"), out_dir / "tube_crop_debug.png")
        save_debug_crop(debug.get("crimp_crop_base64"), out_dir / "crimp_crop_debug.png")


if __name__ == "__main__":
    main()
