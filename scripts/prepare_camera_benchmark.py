"""Freeze evaluation images and publisher annotations before comparing checkpoints."""
import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
from prepare_drinking_waste import perceptual_hash

from wastenot_ai.taxonomy import BROAD_LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/local/broad-waste-v3-batch2"
CACHE = ROOT / "data/local/downloads/camera-benchmark-v1"
COCO_ARCHIVE = ROOT / "data/local/downloads/broad-waste-v1/coco-annotations2017.zip"
COCO_TARGETS = ("bottle", "cup", "cell phone", "remote")
SEED = "camera-replay-v1-20261107"
SCREENING_EXCLUSIONS = {
    76416: "Annotated device is a printed iPod advertisement on a bus; physical-object benchmark scope",
    147518: "Visible cup on right counter lacks target annotation; unsuitable empty-target negative",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rank(identifier):
    return sha((SEED + str(identifier)).encode())


def xyxy(box):
    _, cx, cy, width, height = box
    return [max(0, cx - width / 2), max(0, cy - height / 2),
            min(1, cx + width / 2), min(1, cy + height / 2)]


def local_records(manifest):
    test = [row for row in manifest["records"] if row["split"] == "test"]
    selected = [row for row in test if row["source"] == "TACO"]
    groups = defaultdict(list)
    for row in test:
        if row["source"] == "DrinkingWasteV2":
            groups[row["sequence_group"]].append(row)
    for rows in groups.values():
        selected.extend(sorted(rows, key=lambda row: rank(row["image"]))[:3])
    records = []
    for row in sorted(selected, key=lambda row: row["image"]):
        image = DATA / row["image"]
        checksum = sha(image.read_bytes())
        if checksum != row["prepared_sha256"]:
            raise ValueError("Held-out image changed since dataset preparation")
        is_taco = row["source"] == "TACO"
        records.append({"id": image.stem, "source": row["source"], "split": "test",
            "image": image.relative_to(ROOT).as_posix(), "sha256": checksum,
            "group": row.get("sequence_group") or row["group"], "phash": int(row["phash"]),
            "evaluated_classes": list(BROAD_LABELS) if is_taco else ["bottle", "can"],
            "truth": [{"label": BROAD_LABELS[box[0]], "bbox": xyxy(box)} for box in row["boxes"]],
            "annotation_source": "publisher boxes mapped by existing Broad Waste pipeline",
            "rights": row.get("rights", {"license": "CC0 publisher / MIT acknowledged TrashNet", "source": "DrinkingWasteV2"}),
            "original_source_id": row.get("source_id", row.get("image_id"))})
    return records


def fetch_coco_image(client, image):
    destination = CACHE / image["file_name"]
    if not destination.exists():
        url = "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + image["file_name"]
        response = client.get(url)
        response.raise_for_status()
        if len(response.content) > 5 * 1024 * 1024:
            raise ValueError("COCO image exceeds local media limit")
        destination.write_bytes(response.content)
    with Image.open(destination) as opened:
        if opened.size != (image["width"], image["height"]):
            raise ValueError("COCO image dimensions disagree with annotations")
        thumbnail = opened.convert("RGB")
        thumbnail.thumbnail((960, 960))
        image_hash = perceptual_hash(thumbnail)
        thumbnail.close()
    return destination, image_hash


def coco_records(coco, existing, blocked_hashes):
    names = {row["id"]: row["name"] for row in coco["categories"]}
    licenses = {row["id"]: row for row in coco["licenses"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    candidates = [image for image in coco["images"]
                  if image["license"] in {4, 5} and image["id"] not in SCREENING_EXCLUSIONS]
    candidates.sort(key=lambda image: rank(image["id"]))
    selected_ids = set()
    selected_hashes = [row["phash"] for row in existing]
    rejected = [{"id": ident, "reason": reason} for ident, reason in SCREENING_EXCLUSIONS.items()]
    records = []
    quotas = [("cell phone", 20), ("remote", 20), ("bottle", 10), ("cup", 10), ("empty-targets", 10)]
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for target, limit in quotas:
            accepted = 0
            for image in candidates:
                if accepted >= limit:
                    break
                ident = image["id"]
                if ident in selected_ids:
                    continue
                anns = annotations[ident]
                target_anns = [ann for ann in anns if names[ann["category_id"]] in COCO_TARGETS]
                if target == "empty-targets":
                    if target_anns:
                        continue
                elif not any(names[ann["category_id"]] == target for ann in target_anns):
                    continue
                if any(ann.get("iscrowd") for ann in target_anns):
                    continue
                path, image_hash = fetch_coco_image(client, image)
                if any((image_hash ^ other).bit_count() <= 5 for other in blocked_hashes + selected_hashes):
                    rejected.append({"id": ident, "reason": "near_duplicate_training_validation_or_selected"})
                    selected_ids.add(ident)
                    continue
                truth = []
                for annotation in target_anns:
                    x, y, width, height = annotation["bbox"]
                    truth.append({"label": names[annotation["category_id"]], "bbox": [
                        max(0, x / image["width"]), max(0, y / image["height"]),
                        min(1, (x + width) / image["width"]), min(1, (y + height) / image["height"])]})
                records.append({"id": "COCO-val-" + str(ident), "source": "COCO2017-val", "split": "val2017",
                    "image": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes()),
                    "group": "COCO-val-" + str(ident), "phash": image_hash, "selection_stratum": target,
                    "evaluated_classes": list(COCO_TARGETS), "truth": truth,
                    "annotation_source": "COCO Consortium instances_val2017.json",
                    "rights": {**licenses[image["license"]], "source_page": image.get("flickr_url"),
                               "download_url": "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + image["file_name"]},
                    "original_source_id": ident})
                selected_hashes.append(image_hash)
                selected_ids.add(ident)
                accepted += 1
                print(f"{target} {accepted}/{limit}", flush=True)
            if accepted != limit:
                raise ValueError(f"Insufficient eligible images for {target}")
    return records, rejected


def contact_sheets(records):
    output = ROOT / "artifacts/camera-benchmark-curation"
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), 20):
        sheet = Image.new("RGB", (1400, 1100), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(records[start:start + 20]):
            with Image.open(ROOT / row["image"]) as opened:
                image = opened.convert("RGB")
            image.thumbnail((270, 235))
            left, top = (index % 5) * 280, (index // 5) * 275
            sheet.paste(image, (left, top))
            for truth in row["truth"]:
                x1, y1, x2, y2 = truth["bbox"]
                draw.rectangle((left + x1 * image.width, top + y1 * image.height,
                                left + x2 * image.width, top + y2 * image.height), outline="red", width=2)
            draw.text((left, top + 238), row["id"], fill="black")
            draw.text((left, top + 252), ", ".join(sorted({box["label"] for box in row["truth"]}))[:42], fill="black")
            image.close()
        sheet.save(output / f"annotations-{start // 20 + 1}.jpg")
        sheet.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/camera-replay-v1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Frozen manifest already exists. Choose a new version or reuse it for evaluation.")
    source = json.loads((DATA / "manifest.json").read_text("utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(COCO_ARCHIVE) as archive:
        raw_annotations = archive.read("annotations/instances_val2017.json")
    records = local_records(source)
    blocked = [int(row["phash"]) for row in source["records"] if row["split"] != "test"]
    extra, rejected = coco_records(json.loads(raw_annotations), records, blocked)
    records.extend(extra)
    result = {"version": "camera-replay-v1", "usage": "EVALUATION_ONLY", "kind": "public_camera_proxy_replay",
        "seed": SEED, "target_classes": list(BROAD_LABELS) + ["cell phone", "remote"],
        "source_manifest_sha256": sha((DATA / "manifest.json").read_bytes()),
        "coco_val_annotations_sha256": sha(raw_annotations),
        "selection": "all 32 held-out TACO images; 3 hash-ranked images per DrinkingWaste test sequence; hash-ranked COCO val quotas before inference",
        "duplicate_check": "COCO candidate pHash Hamming distance >5 from Broad Waste train/val and selected images",
        "annotation_review": "publisher annotations; Codex contact-sheet screening; no independent human bbox review",
        "limitations": ["Public replay approximates browser preprocessing; local physical-camera capture holdout is pending.",
            "COCO val may have informed pretrained model development; it is not a new unseen deployment population.",
            "Per-image evaluated_classes restrict scoring to classes covered by that source's annotations.",
            "DW frames share physical objects and sessions; image counts exceed independent object counts.",
            "TACO rare classes have limited support; rankings describe this frozen sample."],
        "records": records, "rejected": rejected,
        "summary": {"images": len(records), "by_source": dict(Counter(row["source"] for row in records)),
                    "truth_instances": dict(Counter(box["label"] for row in records for box in row["truth"]))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    contact_sheets(records)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
