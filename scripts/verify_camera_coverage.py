"""Replay an image through camera capture and upload; save metadata, never photo copies."""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--photo-region", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--url", default="http://127.0.0.1:8098")
    parser.add_argument("--object-key", default="cell_phone")
    parser.add_argument("--output", type=Path, default=Path("artifacts/camera-coverage.json"))
    args = parser.parse_args()
    with Image.open(args.image) as source:
        photo = source.convert("RGB")
    if args.photo_region:
        original = photo
        photo = original.crop(tuple(args.photo_region))
        original.close()
    data = BytesIO()
    photo.save(data, format="JPEG", quality=90)
    photo.close()
    image_url = "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script(
            """Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {value: async () => {
          const image = new Image(); image.src = """
            + json.dumps(image_url)
            + """;
          await image.decode();
          const canvas = document.createElement('canvas');
          canvas.width = image.width; canvas.height = image.height;
          const stream = canvas.captureStream(10);
          canvas.getContext('2d').drawImage(image, 0, 0);
          return stream;
        }});"""
        )
        page.goto(args.url, wait_until="networkidle")
        expect(page.locator("#model-coverage")).to_contain_text("Sampah v3 + benda umum")
        expect(page.locator("#capture")).to_be_enabled()
        page.locator("#capture").click()
        expect(page.locator("#detect")).to_be_visible()
        assert page.evaluate("document.querySelector('#camera').srcObject === null")
        with page.expect_response(lambda r: r.url.endswith("/demo/api/detect")) as capture_response:
            page.locator("#detect").click()
        captured = capture_response.value.json()["data"]
        selectable = [obj for obj in captured["objects"] if obj["supported_for_review"]]
        assert selectable, "Captured photo still has zero selectable candidates"
        page.locator(".bbox:not(.unsupported)").first.click()
        page.locator("#object-key").select_option(args.object_key)
        page.locator("#correct-detection").click()
        page.locator("#provider").select_option("fixture")
        page.locator("#recycling-access").select_option("YES")
        page.locator("#confirmed").check()
        with page.expect_response(lambda r: r.url.endswith("/demo/api/recommend")) as response:
            page.locator("#recommend").click()
        recommendation = response.value.json()["data"]
        expect(page.locator("#method")).to_have_text("Recycle")
        assert recommendation["special_handling"] and recommendation["saved_log"] is False
        page.locator("#file").set_input_files(
            {"name": "replay.jpg", "mimeType": "image/jpeg", "buffer": data.getvalue()}
        )
        with page.expect_response(lambda r: r.url.endswith("/demo/api/detect")) as upload_response:
            page.locator("#detect").click()
        uploaded = upload_response.value.json()["data"]
        captured_labels = {obj["raw_class"] for obj in selectable}
        uploaded_labels = {obj["raw_class"] for obj in uploaded["objects"] if obj["supported_for_review"]}
        assert captured_labels == uploaded_labels, (captured_labels, uploaded_labels)
        assert not errors, errors
        report = {
            "camera": "canvas stream replay; physical camera not accessed",
            "photos_saved": False,
            "live_gemini": False,
            "model_version": captured["model"]["model_version"],
            "camera_selectable": [{"label": obj["raw_class"], "score": obj["confidence"]} for obj in selectable],
            "upload_selectable_labels": sorted(uploaded_labels),
            "manual_correction": args.object_key,
            "recommendation": recommendation["recommended_method"],
            "special_handling": True,
            "browser_errors": errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
