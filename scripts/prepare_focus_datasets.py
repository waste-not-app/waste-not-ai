"""Acquire and prepare cardboard and general-object confusion training supplements."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import httpx
from PIL import Image, ImageDraw
import yaml

import expand_broad_waste as expansion
from prepare_broad_waste import normalized_box
from prepare_drinking_waste import perceptual_hash
from wastenot_ai.taxonomy import COCO_LABELS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/local/broad-waste-v3-batch2"
CACHE = ROOT / "data/local/downloads/focus-v4"
BENCHMARK = ROOT / "benchmarks/camera-replay-v1.json"
GENERAL = ROOT / "data/local/general-confusion-v1"
WASTE = ROOT / "data/local/broad-waste-v4"
TARGETS = ("cell phone", "remote", "refrigerator")
SEED = "focus-v4-20261107"


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def rank(value):
    return hashlib.sha256((SEED + str(value)).encode()).hexdigest()


def frozen_images():
    base = read_json(BASE / "manifest.json")
    benchmark = read_json(BENCHMARK)
    return base, benchmark


def acquire_cardboard(max_requests):
    base, benchmark = frozen_images()
    taco = read_json(expansion.SOURCE / "taco.json")
    plan = expansion.make_plan(base, taco, 300)
    deferred = {row["source_id"] for row in read_json(BASE / "review-snapshot.json")
                if row["decision"] != "APPROVED"}
    plan["candidates"] = [row for row in plan["candidates"] if row["source_id"] not in deferred
                          and any(box[0] == 5 for box in row["boxes"]) and "flickr" in row["original_url"]]
    plan.update(baseline_manifest_sha256=expansion.digest(BASE / "manifest.json"),
                source_metadata_sha256=expansion.digest(expansion.SOURCE / "taco.json"),
                benchmark_sha256=expansion.digest(BENCHMARK))
    expansion.write_json(CACHE / "cardboard/plan.json", plan)
    state = expansion.acquire(plan, CACHE / "cardboard", max_requests)
    print(json.dumps({"cardboard_candidates": len(plan["candidates"]), "acquired": len(state["records"]),
                      "status": state["status"], "last_error": state.get("last_error")}))
    make_sheets(state["records"], CACHE / "cardboard/sheets", expansion.LABELS, "cardboard")


def make_sheets(records, destination, labels, prefix):
    destination.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), 20):
        canvas = Image.new("RGB", (1400, 1100), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(records[start:start + 20]):
            with Image.open(row["cached_path"]) as opened:
                image = opened.convert("RGB")
            image.thumbnail((272, 232))
            left, top = index % 5 * 280, index // 5 * 275
            canvas.paste(image, (left, top))
            targets = []
            for cls, x, y, width, height in row["boxes"]:
                label = labels[int(cls)]
                if prefix == "general" and label not in TARGETS:
                    continue
                targets.append(label)
                draw.rectangle((left + (x-width/2)*image.width, top + (y-height/2)*image.height,
                                left + (x+width/2)*image.width, top + (y+height/2)*image.height), outline="red", width=2)
            draw.text((left, top+236), str(row["source_id"]), fill="black")
            draw.text((left, top+250), ",".join(sorted(set(targets)))[:43], fill="black")
            image.close()
        canvas.save(destination / f"{prefix}-{start//20+1:02}.jpg")
        canvas.close()


def coco_candidates(data, split, excluded_ids):
    categories = {row["id"]: row["name"] for row in data["categories"]}
    if set(categories.values()) != set(COCO_LABELS):
        raise ValueError("COCO category taxonomy differs from native 80 labels")
    annotations = defaultdict(list)
    for row in data["annotations"]:
        annotations[row["image_id"]].append(row)
    licenses = {row["id"]: row for row in data["licenses"]}
    result = []
    for image in data["images"]:
        if image["id"] in excluded_ids or image["license"] not in {4, 5}:
            continue
        anns = annotations[image["id"]]
        if not anns or any(row.get("iscrowd") for row in anns):
            continue
        if max(image["width"], image["height"]) > 4096:
            continue
        try:
            boxes = [normalized_box(row["bbox"], image["width"], image["height"],
                                    COCO_LABELS.index(categories[row["category_id"]])) for row in anns]
        except ValueError:
            continue
        result.append({"source": "COCO2017", "source_id": str(image["id"]), "source_split": split,
            "group": f"COCO-{split}-{image['id']}", "filename": image["file_name"],
            "width": image["width"], "height": image["height"], "boxes": boxes,
            "rights": {"accepted": True, "license_url": licenses[image["license"]]["url"],
                       "source_page": image.get("flickr_url") or image["coco_url"],
                       "license_name": licenses[image["license"]]["name"],
                       "verification": "image-specific publisher COCO license metadata"},
            "url": f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{image['file_name']}"})
    return sorted(result, key=lambda row: rank(row["source_id"]))


def select_candidates(rows, quotas):
    selected, used = [], set()
    for label, count in quotas:
        cls = COCO_LABELS.index(label) if label in TARGETS else None
        candidates = [row for row in rows if row["source_id"] not in used and
                      (any(int(box[0]) == cls for box in row["boxes"]) if cls is not None else
                       all(COCO_LABELS[int(box[0])] not in TARGETS for box in row["boxes"]))]
        # Half of each target stratum emphasizes larger objects; the rest retains small/occluded scenes.
        if cls is not None:
            large = [row for row in candidates if max((box[3]*box[4] for box in row["boxes"] if int(box[0]) == cls), default=0) >= .03]
            order = large[:count//2] + candidates
        else:
            order = candidates
        accepted = 0
        for row in order:
            if row["source_id"] in used:
                continue
            selected.append({**row, "selection_stratum": label})
            used.add(row["source_id"])
            accepted += 1
            if accepted == count:
                break
        if accepted < count:
            raise ValueError(f"Insufficient COCO candidates for {label}: {accepted}/{count}")
    return selected


def download_general(row):
    path = CACHE / "general/images" / (row["source_split"] + "-" + row["filename"])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.get(row["url"])
                response.raise_for_status()
            if len(response.content) > 5*1024*1024:
                raise ValueError("Image exceeds media limit")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(response.content)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]) or image.getexif().get(274, 1) != 1:
                raise ValueError("Image dimensions or orientation disagree with annotations")
            image_hash = perceptual_hash(image.convert("RGB"))
        return {**row, "cached_path": str(path), "raw_sha256": expansion.digest(path), "phash": image_hash}
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return {"source_id": row["source_id"], "source_split": row["source_split"], "rejected": str(exc)}


def acquire_general():
    base, benchmark = frozen_images()
    excluded_train = {int(row["source_id"]) for row in base["records"] if row["source"] == "COCO2017"}
    excluded_val = {int(row["original_source_id"]) for row in benchmark["records"] if row["source"] == "COCO2017-val"}
    archive_path = expansion.SOURCE / "coco-annotations2017.zip"
    selected = []
    with zipfile.ZipFile(archive_path) as archive:
        for split, excluded, quotas in (
            ("train2017", excluded_train, [("cell phone",220),("remote",120),("refrigerator",80),("replay",80)]),
            ("val2017", excluded_val, [("cell phone",30),("remote",12),("refrigerator",20),("replay",25)])):
            raw = archive.read(f"annotations/instances_{split}.json")
            selected.extend(select_candidates(coco_candidates(json.loads(raw), split, excluded), quotas))
    expansion.write_json(CACHE / "general/plan.json", {"selected": selected,
        "baseline_sha256": expansion.digest(BASE / "manifest.json"), "benchmark_sha256": expansion.digest(BENCHMARK),
        "annotations_archive_sha256": expansion.digest(archive_path)})
    blocked = [int(row["phash"]) for row in base["records"] + benchmark["records"]]
    accepted, rejected, seen = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(download_general, selected)):
            if "rejected" in result:
                rejected.append(result)
            elif any((result["phash"] ^ value).bit_count() <= 5 for value in blocked + seen):
                rejected.append({"source_id": result["source_id"], "source_split": result["source_split"], "rejected": "near_duplicate_reserved_or_selected"})
            else:
                accepted.append(result)
                seen.append(result["phash"])
            if (index+1) % 25 == 0:
                expansion.write_json(CACHE / "general/acquisition.json", {"records": accepted, "rejected": rejected})
                print(f"General objects {index+1}/{len(selected)}; accepted {len(accepted)}", flush=True)
    expansion.write_json(CACHE / "general/acquisition.json", {"records": accepted, "rejected": rejected})
    make_sheets(accepted, CACHE / "general/sheets", COCO_LABELS, "general")
    print(json.dumps({"general_images": len(accepted), "rejected": len(rejected),
                      "target_instances": dict(Counter(COCO_LABELS[int(box[0])] for row in accepted for box in row["boxes"] if COCO_LABELS[int(box[0])] in TARGETS))}))


def acquire_openimages():
    base, benchmark = frozen_images()
    cache = CACHE / "openimages"
    cache.mkdir(parents=True, exist_ok=True)
    metadata_urls = {
        "classes.csv": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
        "boxes.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "images.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    }
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for filename, url in metadata_urls.items():
            path = cache / filename
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
    metadata_hashes = {filename: expansion.digest(cache / filename) for filename in metadata_urls}
    metadata = {row["ImageID"]: row for row in csv.DictReader((cache / "images.csv").open(encoding="utf-8"))}
    annotations = defaultdict(list)
    for row in csv.DictReader((cache / "boxes.csv").open(encoding="utf-8")):
        if row["LabelName"] in {"/m/050k8", "/m/025dyy"}:
            annotations[row["ImageID"]].append(row)
    blocked = [int(row["phash"]) for row in base["records"] + benchmark["records"]]
    selected = []
    for ident, boxes in sorted(annotations.items(), key=lambda pair: rank(pair[0])):
        meta = metadata[ident]
        if meta["License"] != "https://creativecommons.org/licenses/by/2.0/" or meta["Rotation"] not in {"", "0", "0.0"}:
            continue
        if any(box["IsGroupOf"] != "0" or box["IsDepiction"] != "0" for box in boxes):
            continue
        if not any((float(box["XMax"])-float(box["XMin"]))*
                   (float(box["YMax"])-float(box["YMin"])) >= .15 for box in boxes):
            continue
        labels = {box["LabelName"] for box in boxes}
        if len(labels) != 1:
            continue
        role = "phone" if next(iter(labels)) == "/m/050k8" else "box"
        path = cache / "images" / (ident + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://open-images-dataset.s3.amazonaws.com/validation/{ident}.jpg"
        if not path.exists():
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
            path.write_bytes(response.content)
        with Image.open(path) as image:
            image_hash = perceptual_hash(image.convert("RGB"))
        if any((image_hash ^ value).bit_count() <= 5 for value in blocked):
            continue
        blocked.append(image_hash)
        cls = COCO_LABELS.index("cell phone") if role == "phone" else 5
        normalized = []
        for box in boxes:
            x1, y1, x2, y2 = (float(box[key]) for key in ("XMin", "YMin", "XMax", "YMax"))
            normalized.append([cls, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
        selected.append({"source": "OpenImages", "source_id": ident, "source_split": "validation",
            "intended_split": "train", "role": role, "group": "OpenImages-" + ident,
            "boxes": normalized, "cached_path": str(path), "raw_sha256": expansion.digest(path), "phash": image_hash,
            "rights": {"accepted": True, "license_url": meta["License"], "source_page": meta["OriginalLandingURL"],
                "creator_page": meta["AuthorProfileURL"], "author": meta["Author"],
                "verification": "image-specific publisher Open Images metadata"},
            "source_annotation_class": "Mobile phone" if role == "phone" else "Box",
            "annotation_review_required": "isolation and completeness; Box also needs visually confirmed cardboard material",
            "url": url, "title": meta["Title"],
            "source_metadata_sha256": metadata_hashes})
        if len(selected) % 20 == 0:
            expansion.write_json(cache / "acquisition.json", {"records": selected})
            print(f"Open Images candidates {len(selected)}", flush=True)
    expansion.write_json(cache / "acquisition.json", {"records": selected})
    for role, labels in (("phone", COCO_LABELS), ("box", expansion.LABELS)):
        make_sheets([row for row in selected if row["role"] == role], cache / "sheets", labels, role)
    print(json.dumps(dict(Counter(row["role"] for row in selected))))


def build_cardboard():
    base, benchmark = frozen_images()
    state = read_json(CACHE / "cardboard/acquisition.json")
    review = read_json(CACHE / "cardboard/review.json")
    if (CACHE / "openimages/review.json").exists():
        extra_review = read_json(CACHE / "openimages/review.json")
        approved_boxes = {row["source_id"] for row in extra_review
                          if row["decision"] == "APPROVED" and row.get("role") == "box"}
        extra = [row for row in read_json(CACHE / "openimages/acquisition.json")["records"]
                 if row["source_id"] in approved_boxes]
        state = {**state, "records": state["records"] + extra}
        review.extend(row for row in extra_review if row["source_id"] in approved_boxes)
    benchmark_hashes = [int(row["phash"]) for row in benchmark["records"]]
    excluded = []
    for row in state["records"]:
        with Image.open(row["cached_path"]) as opened:
            image_hash = perceptual_hash(opened.convert("RGB"))
        if any((image_hash ^ value).bit_count() <= 5 for value in benchmark_hashes):
            excluded.append(row["source_id"])
    reviews = [dict(row, decision="DEFERRED", reason="near frozen benchmark") if row["source_id"] in excluded else row for row in review]
    expansion.build_dataset(base, state, reviews, WASTE, 600, base_dir=BASE, version="broad-waste-v4")
    result = read_json(WASTE / "manifest.json")
    result["benchmark_sha256"] = expansion.digest(BENCHMARK)
    result["benchmark_near_duplicates_excluded"] = excluded
    expansion.write_json(WASTE / "manifest.json", result)
    notices = WASTE / "DATASET_NOTICES.md"
    with notices.open("a", encoding="utf-8") as handle:
        handle.write("\n\nOpen Images, Google/CVDF, https://storage.googleapis.com/openimages/web/download.html .\n"
                     "Included Open Images photos carry publisher-declared CC BY 2.0 metadata.\n"
                     "Per-photo authors, original URLs, rights and reviewed Box-to-cardboard mapping are in manifest.json.\n")
    expansion.write_json(ROOT / "evaluations/focus-v4-cardboard-build.json", {key: value for key, value in result.items() if key not in {"records","rejected"}})
    print(json.dumps({"new_images":result["new_images"], "active_instances":result["active_instances_per_split"]},indent=2))


def assign_general_splits(rows, waste_records):
    heldout = defaultdict(list)
    for row in rows:
        if row["source_split"] == "val2017":
            heldout[row["selection_stratum"]].append(row)
    val_ids = set()
    for group in heldout.values():
        group.sort(key=lambda row: rank(row["source_id"]))
        val_ids.update(row["source_id"] for row in group[::2])
    assigned, rejected = [], []
    for row in rows:
        if row["source_split"] == "train2017" or row.get("intended_split") == "train":
            split = "train"
        else:
            split = "val" if row["source_id"] in val_ids else "test"
        conflicts = [other for other in waste_records if other["split"] != split
                     and (int(other["phash"]) ^ int(row["phash"])).bit_count() <= 5]
        if conflicts:
            rejected.append({"source_id": row["source_id"], "reason": "cross_model_split_conflict",
                             "waste_image": conflicts[0]["image"], "waste_split": conflicts[0]["split"]})
        else:
            assigned.append({**row, "split": split})
    return assigned, rejected


def build_general():
    acquisition = read_json(CACHE / "general/acquisition.json")
    plan = read_json(CACHE / "general/plan.json")
    if plan["benchmark_sha256"] != expansion.digest(BENCHMARK):
        raise ValueError("Frozen benchmark changed since acquisition")
    if plan["baseline_sha256"] != expansion.digest(BASE / "manifest.json"):
        raise ValueError("Broad Waste source snapshot changed since acquisition")
    review_rows = read_json(CACHE / "general/review.json")
    reviews = {row["source_id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError("Duplicate identities in general-object review")
    if (CACHE / "openimages/review.json").exists():
        extra_reviews = read_json(CACHE / "openimages/review.json")
        extra_by_id = {row["source_id"]: row for row in extra_reviews if row["decision"] == "APPROVED"}
        for row in read_json(CACHE / "openimages/acquisition.json")["records"]:
            review = extra_by_id.get(row["source_id"])
            if review is None:
                continue
            if review["role"] == "phone":
                acquisition["records"].append(row)
            elif review.get("general_negative"):
                acquisition["records"].append({**row, "boxes": [], "kind": "VERIFIED_NEGATIVE",
                    "source_boxes": row["boxes"], "selection_stratum": "cardboard_counterexample"})
        reviews.update({row["source_id"]: row for row in extra_reviews})
    rows = [row for row in acquisition["records"]
            if reviews.get(row["source_id"], {}).get("decision") == "APPROVED"]
    if not rows or GENERAL.exists() or GENERAL.with_suffix(".building").exists():
        raise ValueError("Need approved photos and a fresh output path")
    from prepare_drinking_waste import parse_boxes
    for row in rows:
        review = reviews[row["source_id"]]
        if review["raw_sha256"] != row["raw_sha256"] or expansion.digest(Path(row["cached_path"])) != row["raw_sha256"]:
            raise ValueError("Review/image checksum mismatch")
        if not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError("Review provenance missing")
        if row["boxes"]:
            parse_boxes("\n".join(" ".join(map(str, box)) for box in row["boxes"]), len(COCO_LABELS))
        elif row.get("kind") != "VERIFIED_NEGATIVE" or not review.get("general_negative"):
            raise ValueError("Empty training labels require explicit reviewed negative status")
    reserved = [int(row["phash"]) for row in read_json(BENCHMARK)["records"]]
    seen = []
    for row in rows:
        if any((int(row["phash"]) ^ value).bit_count() <= 5 for value in reserved + seen):
            raise ValueError("Near duplicate in approved records or frozen benchmark")
        seen.append(int(row["phash"]))
    waste = read_json(WASTE / "manifest.json")
    rows, split_rejections = assign_general_splits(rows, waste["records"])
    if not rows:
        raise ValueError("No approved records remain after cross-model split checks")
    staging = GENERAL.with_suffix(".building")
    records = []
    for row in rows:
        split = row["split"]
        relative = f"images/{split}/{row['source']}-{row['source_id']}.jpg"
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["cached_path"], destination)
        label = staging / "labels" / split / (destination.stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(f"{int(box[0])} " + " ".join(f"{value:.8f}" for value in box[1:]) + "\n"
                                 for box in row["boxes"]), encoding="utf-8")
        records.append({**row, "split": split, "image": relative,
                        "prepared_sha256": expansion.digest(destination),
                        "visual_review": reviews[row["source_id"]]})
    for split in expansion.SPLITS:
        (staging / (split + ".txt")).write_text(
            "".join((GENERAL / row["image"]).as_posix() + "\n" for row in records if row["split"] == split),
            encoding="utf-8")
    configuration = {"path": GENERAL.as_posix(), "train": "train.txt", "val": "val.txt", "test": "test.txt",
                     "names": dict(enumerate(COCO_LABELS))}
    (staging / "data.yaml").write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    instance_counts = {}
    for split in expansion.SPLITS:
        counts = Counter(int(box[0]) for row in records if row["split"] == split for box in row["boxes"])
        instance_counts[split] = {name: counts[index] for index, name in enumerate(COCO_LABELS)}
    manifest = {"version": "general-confusion-v1", "labels": list(COCO_LABELS), "records": records,
        "focus_classes": list(TARGETS), "status": "EXPERIMENTAL", "benchmark_sha256": expansion.digest(BENCHMARK),
        "source_plan_sha256": expansion.digest(CACHE / "general/plan.json"),
        "images_per_split": dict(Counter(row["split"] for row in records)),
        "instances_per_split": instance_counts,
        "negative_images": sum(row.get("kind") == "VERIFIED_NEGATIVE" for row in records),
        "waste_manifest_sha256": expansion.digest(WASTE / "manifest.json"),
        "cross_model_split_rejections": split_rejections,
        "limitations": ["COCO train photos were available in general-model pretraining; this is task-focused resampling for fine-tuning.",
            "COCO val photos supply fresh fine-tuning val/test, with frozen camera benchmark and broad v3 photographs excluded.",
            "Approved Open Images phone and cardboard negative supplements use project train split.",
            "Open Images target boxes have not been independently audited for completeness of every COCO class.",
            "Modern three-camera rear phone views remain underrepresented.",
            "COCO image IDs and pHash approximate independent physical scenes.",
            "Visual screening is by Codex contact sheets; independent human review remains pending."]}
    expansion.write_json(staging / "manifest.json", manifest)
    expansion.write_json(staging / "review-snapshot.json", list(reviews.values()))
    expansion.write_json(staging / "source-plan-snapshot.json", plan)
    notices = ("# General-object confusion subset\n\n"
        "COCO Consortium, Lin et al., Microsoft COCO: Common Objects in Context. https://cocodataset.org/\n"
        "COCO annotations CC BY 4.0. Image-specific licenses and source URLs are retained in manifest.json.\n\n"
        "Open Images, Google/CVDF, https://storage.googleapis.com/openimages/web/download.html .\n"
        "Open Images annotations CC BY 4.0 and included photos publisher-declared CC BY 2.0.\n"
        "Per-photo author, license and original landing URL are retained in manifest.json.\n")
    (staging / "DATASET_NOTICES.md").write_text(notices, encoding="utf-8")
    staging.replace(GENERAL)
    expansion.write_json(ROOT / "evaluations/focus-v4-general-build.json",
                         {key: value for key, value in manifest.items() if key != "records"})
    print(json.dumps({"images_per_split": manifest["images_per_split"], "negative_images": manifest["negative_images"],
        "focus_instances": {split: {label: instance_counts[split][label] for label in TARGETS}
                            for split in expansion.SPLITS}}, indent=2))


def audit_focus_datasets():
    base, benchmark = frozen_images()
    waste = read_json(WASTE / "manifest.json")
    general = read_json(GENERAL / "manifest.json")
    benchmark_digest = expansion.digest(BENCHMARK)
    if waste["benchmark_sha256"] != benchmark_digest or general["benchmark_sha256"] != benchmark_digest:
        raise ValueError("Frozen camera benchmark checksum changed")
    if waste["baseline_manifest_sha256"] != expansion.digest(BASE / "manifest.json"):
        raise ValueError("Inherited Broad Waste manifest changed")
    if general["waste_manifest_sha256"] != expansion.digest(WASTE / "manifest.json"):
        raise ValueError("General-object package references a different waste dataset")

    inherited = {row["image"]: row for row in base["records"]}
    current = {row["image"]: row for row in waste["records"]}
    preserved_fields = ("prepared_sha256", "split", "boxes", "group", "phash")
    for name, original in inherited.items():
        if name not in current or any(current[name][key] != original[key] for key in preserved_fields):
            raise ValueError(f"Inherited photograph or annotation changed: {name}")

    reserved_matches = []
    for package, records in (("waste", waste["records"]), ("general", general["records"])):
        for row in records:
            for reserved in benchmark["records"]:
                if (int(row["phash"]) ^ int(reserved["phash"])).bit_count() > 5:
                    continue
                # The frozen replay deliberately reuses existing v3 test photographs.
                if package != "waste" or row["image"] not in inherited or row["split"] != "test":
                    raise ValueError(f"Camera benchmark leaked into new data: {row['image']}")
                reserved_matches.append({"image": row["image"], "benchmark_id": reserved["id"]})

    shared = []
    for row in general["records"]:
        for other in waste["records"]:
            same_image = row["prepared_sha256"] == other["prepared_sha256"]
            near_image = (int(row["phash"]) ^ int(other["phash"])).bit_count() <= 5
            same_group = row["group"] == other["group"]
            if not (same_image or near_image or same_group):
                continue
            if row["split"] != other["split"]:
                raise ValueError(f"Cross-model split leakage: {row['image']} / {other['image']}")
            shared.append({"general_image": row["image"], "waste_image": other["image"],
                           "split": row["split"], "identical_sha256": same_image})

    result = {"status": "PASS", "benchmark_sha256": benchmark_digest,
              "inherited_images_preserved": len(inherited),
              "new_waste_images": len(current) - len(inherited),
              "general_images": len(general["records"]),
              "existing_waste_test_benchmark_matches": reserved_matches,
              "cross_model_shared_or_near_images": shared,
              "new_data_benchmark_matches": 0, "cross_model_split_conflicts": 0,
              "scope": "Manifest identity and proxy-group/pHash leakage checks. Pair with verify_broad_data.py for file and label checks."}
    expansion.write_json(ROOT / "evaluations/focus-v4-leakage-audit.json", result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"existing_waste_test_benchmark_matches", "cross_model_shared_or_near_images"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["acquire-cardboard", "acquire-general", "acquire-openimages",
                                          "build-cardboard", "build-general", "audit"])
    parser.add_argument("--max-requests", type=int, default=180)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.stage == "acquire-cardboard":
        acquire_cardboard(args.max_requests)
    elif args.stage == "acquire-general":
        acquire_general()
    elif args.stage == "acquire-openimages":
        acquire_openimages()
    elif args.stage == "build-cardboard":
        build_cardboard()
    elif args.stage == "build-general":
        build_general()
    else:
        audit_focus_datasets()


if __name__ == "__main__":
    main()
