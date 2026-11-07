"""Render downloaded detection images with original target boxes for local screening."""
import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from prepare_broad_waste import LABELS

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("cyan", "orange", "yellow", "lime", "magenta", "red", "dodgerblue", "white", "cyan", "yellow")


def render(records, output):
    output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    index = []
    for page in range((len(records) + 11) // 12):
        sheet = Image.new("RGB", (1320, 1140), "#222222")
        draw = ImageDraw.Draw(sheet)
        page_rows = records[page * 12:(page + 1) * 12]
        for k, row in enumerate(page_rows):
            x, y = k % 4 * 330, k // 4 * 380
            with Image.open(row["cached_path"]) as opened:
                im = opened.convert("RGB")
            im.thumbnail((320, 300))
            boxes = ImageDraw.Draw(im)
            for cls, xc, yc, w, h in row["boxes"]:
                cls = int(cls)
                x1, y1, x2, y2 = (xc-w/2)*im.width, (yc-h/2)*im.height, (xc+w/2)*im.width, (yc+h/2)*im.height
                boxes.rectangle((x1, y1, x2, y2), outline=COLORS[cls], width=2)
                boxes.text((x1, max(0, y1 - 15)), LABELS[cls], font=font,
                           fill=COLORS[cls], stroke_width=1, stroke_fill="black")
            draw.text((x+3, y+2), "TACO " + row["source_id"] + " / " + row["group"], font=font, fill="white")
            sheet.paste(im, (x+3, y+25))
            caption = ", ".join(LABELS[int(b[0])] for b in row["boxes"])
            for line, text in enumerate(textwrap.wrap(caption, width=39)[:3]):
                draw.text((x+3, y+328+16*line), text, font=font, fill="white")
        target = output / f"audit-{page:02}.jpg"
        sheet.save(target, quality=95)
        index.append({"sheet": target.name, "source_ids": [r["source_id"] for r in page_rows]})
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, default=ROOT / "data/local/downloads/broad-waste-v3/acquisition.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.acquisition.read_text("utf-8"))["records"][args.start:]
    print(json.dumps(render(rows, args.output), indent=2))


if __name__ == "__main__":
    main()
