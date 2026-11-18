"""Audit publisher samples and build the manually reviewed RealWaste research supplement."""
import argparse
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import expand_broad_waste as expansion
import httpx
from PIL import Image
from prepare_broad_waste import normalized_box
from prepare_drinking_waste import perceptual_hash

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/local/downloads/external-focus-v5"
BASE = ROOT / "data/local/broad-waste-v4"
OUTPUT = ROOT / "data/local/broad-waste-v5-realwaste-research"
REVIEW = ROOT / "annotations/realwaste-cardboard-v1.json"
REALWASTE_COMMIT = "156e8e9154eeb163df30fcfd767f55b19633bb06"
FPI_COMMIT = "83c452d1a1e5b1bac88afe1a6834cb9c25e036eb"
FPI_ARCHIVE_SHA256 = "5eea5d8ce0317c3bfabecb29c54eea2845a022bcfd190b4c8ca781b52bf7cb48"


def read(path):
    return json.loads(path.read_text("utf-8"))


def publisher_file(repository, commit, row, destination):
    if not destination.exists():
        response = httpx.get(f"https://raw.githubusercontent.com/{repository}/{commit}/{row['path']}",
                             timeout=60, follow_redirects=True)
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(destination)
    raw = destination.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != row["sha"]:
        raise ValueError(f"Publisher Git blob differs from the pinned tree: {destination}")
    return destination


def acquire():
    CACHE.mkdir(parents=True, exist_ok=True)
    for prefix, repository, commit in (("fpi", "KvCgRv/FPI-Det", FPI_COMMIT),
                                        ("realwaste", "sam-single/realwaste", REALWASTE_COMMIT)):
        tree_path = CACHE / (prefix + "-tree.json")
        if not tree_path.exists():
            response = httpx.get(f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1", timeout=60)
            response.raise_for_status()
            expansion.write_json(tree_path, response.json())
        tree = read(tree_path)
        if tree["sha"] != commit or tree.get("truncated"):
            raise ValueError("Publisher tree is not the pinned complete snapshot")
        files = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
        publisher_file(repository, commit, files["README.md"], CACHE / (prefix + "-README.md"))
        if prefix == "fpi":
            for source, filename in (("LICENSE", "fpi-LICENSE.txt"), ("Phone.yaml", "fpi-Phone.yaml"),
                                     ("annotations/instances_train2017.json", "fpi-train.json"),
                                     ("annotations/instances_val2017.json", "fpi-val.json"),
                                     ("figure_mopho/datasetsample.png", "fpi-datasetsample.png"),
                                     ("phone_train_yolo.rar", "fpi-labels.rar")):
                publisher_file(repository, commit, files[source], CACHE / filename)
        else:
            candidates = [row for name, row in files.items() if name.startswith("RealWaste/Cardboard/")]
            candidates.sort(key=lambda row: hashlib.sha256(("external-focus-v5-20261107" + row["path"]).encode()).hexdigest())

            def download(row):
                path = publisher_file(repository, commit, row, CACHE / "realwaste-images" / Path(row["path"]).name)
                return {**row, "cached_path": str(path.resolve()), "raw_sha256": expansion.digest(path), "source_id": path.stem}

            with ThreadPoolExecutor(max_workers=4) as pool:
                selected = list(pool.map(download, candidates[:60]))
            expansion.write_json(CACHE / "realwaste-acquisition.json", {"commit": commit, "selected": selected})
            print(f"Verified {len(selected)} RealWaste photos against pinned publisher blobs")


def parse_fpi_labels(text):
    boxes, combined_lines = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) % 5 or not all(math.isfinite(value) for value in values):
            raise ValueError("FPI label must contain complete finite five-column records")
        combined_lines += len(values) > 5
        for start in range(0, len(values), 5):
            cls, x, y, width, height = values[start:start + 5]
            if cls not in (0, 1):
                raise ValueError("Unexpected native FPI class")
            boxes.append([int(cls), x, y, width, height])
    return boxes, combined_lines


def phash_neighbors(rows):
    # Six disjoint bands ensure every distance <= 5 pair shares a complete band.
    bands = ((0, 11), (11, 11), (22, 11), (33, 10), (43, 10), (53, 10))
    index = defaultdict(list)
    for i, row in enumerate(rows):
        value = int(row["phash"])
        keys = [(band, value >> shift & ((1 << width) - 1)) for band, (shift, width) in enumerate(bands)]
        candidates = {j for key in keys for j in index[key]}
        for j in sorted(candidates):
            if (value ^ int(rows[j]["phash"])).bit_count() <= 5:
                yield i, j
        for key in keys:
            index[key].append(i)


def audit_pixels(archive_path):
    with archive_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != FPI_ARCHIVE_SHA256:
            raise ValueError("FPI image archive checksum mismatch")
    cache_path = CACHE / "fpi-pixels.json"
    if not cache_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            sources = read(CACHE / "fpi-archive-index.json")["records"]

            def inspect(row):
                raw = archive.read(row["image_member"])
                with Image.open(BytesIO(raw)) as image:
                    image_hash = perceptual_hash(image.convert("RGB"))
                    width, height = image.size
                return {"source_id": row["source_id"], "source_split": row["source_split"],
                        "sha256": hashlib.sha256(raw).hexdigest(), "phash": image_hash, "width": width, "height": height}

            with ThreadPoolExecutor(max_workers=4) as pool:
                pixels = list(pool.map(inspect, sources))
        expansion.write_json(cache_path, {"archive_sha256": FPI_ARCHIVE_SHA256, "records": pixels})
    cache = read(cache_path)
    if cache["archive_sha256"] != FPI_ARCHIVE_SHA256 or len(cache["records"]) != 22879:
        raise ValueError("FPI pixel cache does not match the complete audited archive")
    pixels = cache["records"]
    protected_manifest = (OUTPUT if OUTPUT.exists() else BASE) / "manifest.json"
    protected = read(protected_manifest)["records"]
    protected += read(ROOT / "benchmarks/camera-replay-v1.json")["records"]
    protected += read(ROOT / "data/local/general-confusion-v1/manifest.json")["records"]
    rows = [dict(row, protected=True) for row in protected] + pixels
    cross_source, cross_split, within_split, examples = 0, 0, 0, []
    for i, j in phash_neighbors(rows):
        left, right = rows[i], rows[j]
        if left.get("protected"):
            continue
        if right.get("protected"):
            cross_source += 1
            if len(examples) < 40:
                examples.append({"kind": "protected_dataset_match", "fpi": left["source_id"], "protected_image": right["image"]})
        elif left["source_split"] != right["source_split"]:
            cross_split += 1
            if len(examples) < 40:
                examples.append({"kind": "cross_publisher_split", "a": left["source_id"], "b": right["source_id"],
                                 "a_split": left["source_split"], "b_split": right["source_split"]})
        else:
            within_split += 1
    sha_groups = defaultdict(list)
    for row in pixels:
        sha_groups[row["sha256"]].append(row)
    annotations = {(row["source_id"], row["source_split"]): row["boxes"]
                   for row in read(CACHE / "fpi-archive-index.json")["records"]}
    conflicting_groups = 0
    for group in sha_groups.values():
        signatures = {tuple(sorted(tuple(round(value, 6) for value in box)
                                    for box in annotations[(row["source_id"], row["source_split"])])) for row in group}
        conflicting_groups += len(signatures) > 1
    report = {"archive_sha256": FPI_ARCHIVE_SHA256, "decoded_images": len(pixels),
              "protected_waste_manifest_sha256": expansion.digest(protected_manifest),
              "pixel_cache_sha256": expansion.digest(cache_path), "phash_distance_max": 5,
              "exact_duplicate_photos": sum(len(group) - 1 for group in sha_groups.values()),
              "exact_hash_groups_spanning_splits": sum(len({row["source_split"] for row in group}) > 1 for group in sha_groups.values()),
              "exact_image_groups_with_different_annotations": conflicting_groups,
              "annotation_comparison": "Order-independent boxes rounded to six normalized decimal places",
              "near_pairs_cross_publisher_split": cross_split, "near_pairs_within_publisher_split": within_split,
              "protected_dataset_match_pairs": cross_source, "examples": examples,
              "limitations": ["pHash is a proxy for visual duplication, not proof of physical-session independence.",
                              "Match counts are pairs and may include multiple references to the same protected photograph."]}
    expansion.write_json(ROOT / "evaluations/fpi-det-pixel-audit.json", report)
    print(json.dumps(report, indent=2))


def audit_archive(path):
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if checksum != FPI_ARCHIVE_SHA256:
        raise ValueError("FPI raw archive differs from the audited publisher download")
    records, counts, flags, combined = [], Counter(), Counter(), []
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("FPI archive CRC check failed")
        names = set(archive.namelist())
        for name in sorted(names):
            if not name.endswith(".txt"):
                continue
            split, ident = name.split("/")[1], Path(name).stem
            image_member = f"reorganized_dataset/{split}/images/{ident}.jpg"
            if image_member not in names:
                raise ValueError("FPI label has no paired image")
            boxes, combined_lines = parse_fpi_labels(archive.read(name).decode("utf-8-sig"))
            if combined_lines:
                combined.append(name)
            invalid = []
            for cls, x, y, width, height in boxes:
                counts[(split, cls)] += 1
                excursion = max(0, -(x - width / 2), -(y - height / 2), x + width / 2 - 1, y + height / 2 - 1)
                if width <= 0 or height <= 0 or excursion > 1e-7:
                    flags[(split, cls)] += 1
                    invalid.append({"class": cls, "excursion": excursion, "nonpositive": width <= 0 or height <= 0})
            records.append({"source_id": ident, "source_split": split, "image_member": image_member,
                            "label_member": name, "boxes": boxes, "geometry_flags": invalid,
                            "class0_area": max((box[3] * box[4] for box in boxes if box[0] == 0), default=0)})
        sample = []
        for split, quota in (("train", 40), ("val", 10), ("test", 10)):
            pool = [row for row in records if row["source_split"] == split and not row["geometry_flags"] and row["class0_area"] >= .03]
            pool.sort(key=lambda row: hashlib.sha256(("fpi-audit20261107" + row["source_id"]).encode()).hexdigest())
            for row in pool[:quota]:
                destination = CACHE / "fpi-sample" / (row["source_id"] + ".jpg")
                destination.parent.mkdir(exist_ok=True)
                raw = archive.read(row["image_member"])
                destination.write_bytes(raw)
                sample.append({**row, "cached_path": str(destination.resolve()), "raw_sha256": hashlib.sha256(raw).hexdigest()})
    expansion.write_json(CACHE / "fpi-archive-index.json", {"archive_sha256": checksum, "records": records,
                                                           "reformatted_label_files": combined})
    expansion.write_json(CACHE / "fpi-sample.json", {"records": sample})
    return {"archive_sha256": checksum, "archive_size_bytes": path.stat().st_size, "crc_integrity": "PASS",
            "image_label_pairs": len(records), "images_per_split": dict(Counter(row["source_split"] for row in records)),
            "native_class_counts": {split: {"phone": counts[(split, 0)], "head_face": counts[(split, 1)]} for split in ("train", "val", "test")},
            "geometry_flag_counts": {split: {"phone": flags[(split, 0)], "head_face": flags[(split, 1)]} for split in ("train", "val", "test")},
            "labels_with_combined_box_lines": len(combined), "audit_sample_images": len(sample),
            "sample_selection": "Deterministic 40 train, 10 val, 10 test with valid geometry and phone box area >= 3%"}


def audit_fpi(archive=None):
    results = {}
    filenames = {}
    for split in ("train", "val"):
        data = read(CACHE / f"fpi-{split}.json")
        images = {row["id"]: row for row in data["images"]}
        if len(images) != len(data["images"]):
            raise ValueError("Duplicate image identities within FPI split")
        filenames[split] = {row["file_name"] for row in images.values()}
        counts, geometry = Counter(), Counter()
        phone_images, areas = set(), []
        for box in data["annotations"]:
            image = images[box["image_id"]]
            x, y, width, height = box["bbox"]
            counts[box["category_id"]] += 1
            excursion = max(0, -x, -y, x + width - image["width"], y + height - image["height"])
            if width <= 0 or height <= 0:
                geometry[f"class_{box['category_id']}_nonpositive"] += 1
            elif excursion > 1e-5:
                geometry[f"class_{box['category_id']}_" + ("outside_le1px" if excursion <= 1 else "outside_gt1px")] += 1
            if box["category_id"] == 0:
                phone_images.add(image["file_name"])
                areas.append(width * height / (image["width"] * image["height"]))
        results[split] = {"images": len(images), "annotations": len(data["annotations"]),
                          "class_counts": dict(counts), "geometry_flags": dict(geometry),
                          "images_with_phone": len(phone_images), "images_without_phone": len(images) - len(phone_images),
                          "phone_boxes_le_1pct_area": sum(area <= .01 for area in areas),
                          "phone_boxes_le_3pct_area": sum(area <= .03 for area in areas),
                          "labels_sha256": expansion.digest(CACHE / f"fpi-{split}.json")}
    results.update(source="https://github.com/KvCgRv/FPI-Det", commit=FPI_COMMIT,
                   native_classes={"0": "phone", "1": "face"}, images_acquired=0,
                   status="IMAGES_REQUIRE_AUTHENTICATED_DOWNLOAD",
                   cross_split_filename_overlap=len(filenames["train"] & filenames["val"]),
                   inspected_publisher_collage="figure_mopho/datasetsample.png",
                   published_readme_counts={"images": 22879, "phone_annotations": 10255},
                   accessible_annotation_counts={"images": sum(row["images"] for row in results.values()),
                                                 "phone_annotations": sum(row["class_counts"][0] for row in results.values())},
                   repository_photos="Publisher example collages; the RAR contains training code, not raw dataset photos.",
                   limitations=["Actual photos require the publisher Drive download, currently showing sign-in.",
                                "Pixel-level hash/pHash and physical-session grouping require the raw photos.",
                                "Face boxes must not be mapped to COCO person boxes.",
                                "Geometry flags require image review; no automatic clipping was performed.",
                                "Public train/val annotation totals differ from README totals. Reconcile the raw archive/splits before training."])
    if archive is not None:
        results["raw_archive"] = audit_archive(archive)
        results["status"] = "RAW_ARCHIVE_AND_NATIVE_AUDIT_SAMPLE_AVAILABLE"
        results["images_acquired"] = results["raw_archive"]["image_label_pairs"]
        results["limitations"] = ["Public COCO-converted labels and original YOLO label totals differ; preserve all original five-column records.",
                                  "Original publisher splits can share physical sessions. Image/session-level leakage must be audited before training.",
                                  "Face/head boxes cannot be substituted for COCO person boxes.",
                                  "Native FPI phone/head annotations are incomplete for the full 80-class COCO task.",
                                  "Only the 60-photo audit sample has been visually inspected in this batch."]
    expansion.write_json(ROOT / "evaluations/fpi-det-audit.json", results)
    print(json.dumps(results, indent=2))


def export_fpi_sample(output):
    from prepare_drinking_waste import parse_boxes

    if output.exists():
        raise ValueError("Audit sample output exists. Choose a fresh --output")
    rows = read(CACHE / "fpi-sample.json")["records"]
    reviews = {row["source_id"]: row for row in read(ROOT / "annotations/fpi-det-sample-v1.json")}
    pixels = {row["source_id"]: row for row in read(CACHE / "fpi-pixels.json")["records"]}
    records = []
    for row in rows:
        if expansion.digest(Path(row["cached_path"])) != reviews[row["source_id"]]["raw_sha256"]:
            raise ValueError("FPI review/photo identity mismatch")
        labels = "".join(str(box[0]) + " " + " ".join(format(value, ".17g") for value in box[1:]) + "\n" for box in row["boxes"])
        parse_boxes(labels, 2)
        records.append({**row, "phash": pixels[row["source_id"]]["phash"],
                        "width": pixels[row["source_id"]]["width"], "height": pixels[row["source_id"]]["height"],
                        "image": f"images/{row['source_split']}/{row['source_id']}.jpg", "review": reviews[row["source_id"]]})
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise ValueError("Audit staging folder exists. Choose a fresh --output")
    for row in records:
        destination = staging / row["image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["cached_path"], destination)
        label = staging / "labels" / row["source_split"] / (row["source_id"] + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(str(box[0]) + " " + " ".join(format(value, ".17g") for value in box[1:]) + "\n"
                                 for box in row["boxes"]), encoding="utf-8")
    manifest = {"status": "AUDIT_ONLY_NATIVE_PHONE_FACE", "labels": ["phone", "face"], "records": records,
                "archive_sha256": FPI_ARCHIVE_SHA256, "publisher_splits_are_provisional": True,
                "scope": "Phone target screening only. Full COCO class annotation and physical-session grouping require further review.",
                "review_counts": dict(Counter(row["review"]["decision"] for row in records))}
    expansion.write_json(staging / "manifest.json", manifest)
    (staging / "DATASET_NOTICES.md").write_text("# FPI-Det audit sample\n\n"
        "Publisher KvCgRv/FPI-Det declares MIT. Source https://github.com/KvCgRv/FPI-Det .\n"
        "Original phone/head-face annotations retained. Audit-only scope; native face is not a COCO person label.\n", encoding="utf-8")
    staging.rename(output)
    print(json.dumps({"audit_sample_images": len(records), "status": manifest["status"]}))


def build(output=OUTPUT):
    base = read(BASE / "manifest.json")
    acquisition = read(CACHE / "realwaste-acquisition.json")
    metadata = CACHE / "realwaste-tree.json"
    tree = read(metadata)
    if acquisition["commit"] != REALWASTE_COMMIT or tree["sha"] != REALWASTE_COMMIT or tree.get("truncated"):
        raise ValueError("RealWaste source snapshot differs from the pinned publisher commit")
    source_files = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    publisher_file("sam-single/realwaste", REALWASTE_COMMIT, source_files["README.md"], CACHE / "realwaste-README.md")
    if "CC BY-NC-SA 4.0" not in (CACHE / "realwaste-README.md").read_text("utf-8"):
        raise ValueError("Expected author license evidence is missing")
    reviews = read(REVIEW)
    approved = {row["source_id"]: row for row in reviews if row["decision"] == "APPROVED"}
    reserved = base["records"] + read(ROOT / "benchmarks/camera-replay-v1.json")["records"]
    reserved += read(ROOT / "data/local/general-confusion-v1/manifest.json")["records"]
    blocked_hashes = [int(row["phash"]) for row in reserved]
    rows, excluded = [], []
    for source in acquisition["selected"]:
        review = approved.get(source["source_id"])
        if review is None:
            continue
        path = Path(source["cached_path"])
        publisher_file("sam-single/realwaste", REALWASTE_COMMIT, source_files[source["path"]], path)
        if expansion.digest(path) != review["raw_sha256"]:
            raise ValueError("Manual annotation image checksum mismatch")
        x1, y1, x2, y2 = review["pixel_bbox_xyxy"]
        with Image.open(path) as image:
            width, height = image.size
            image_hash = perceptual_hash(image.convert("RGB"))
        if any((image_hash ^ other).bit_count() <= 5 for other in blocked_hashes):
            excluded.append({"source_id": source["source_id"], "reason": "near existing dataset or benchmark"})
            continue
        blocked_hashes.append(image_hash)
        box = normalized_box([x1, y1, x2 - x1, y2 - y1], width, height, 5)
        rows.append({"source": "RealWaste", "source_id": source["source_id"], "cached_path": str(path),
                     "raw_sha256": source["raw_sha256"], "phash": image_hash, "width": width, "height": height,
                     "group": "RealWaste-cardboard-single-capture-family", "boxes": [box],
                     "annotation_source": "Codex manual pixel XYXY bounding box; publisher supplied only image-level material class",
                     "publisher_git_blob": source["sha"], "publisher_commit": REALWASTE_COMMIT,
                     "rights": {"accepted": True, "accepted_for": "local_noncommercial_research",
                                "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                                "source_page": f"https://github.com/sam-single/realwaste/blob/{REALWASTE_COMMIT}/" + source["path"],
                                "author": "Sam Single, Saeid Iranmanesh, Raad Raad",
                                "verification": "Pinned author README declares CC BY-NC-SA 4.0. UCI lists CC BY 4.0; retain stricter author terms.",
                                "commercial_use": "UNRESOLVED_LICENSE_CONFLICT"}})
    if not rows:
        raise ValueError("No approved new photos remain")
    assigned, _ = expansion.split_new_records(base["records"], rows)
    if len(assigned) != len(rows) or any(row["split"] != "train" for row in assigned):
        raise ValueError("This research supplement must keep the entire new capture family on train")
    state = {"records": rows, "baseline_manifest_sha256": expansion.digest(BASE / "manifest.json"),
             "source_metadata_sha256": expansion.digest(metadata)}
    expansion.build_dataset(base, state, reviews, output, 600, base_dir=BASE,
                            version="broad-waste-v5-realwaste-research", source_metadata_path=metadata,
                            source_notice="RealWaste by Sam Single, Saeid Iranmanesh and Raad Raad. Author repository CC BY-NC-SA 4.0. Research batch retains these terms despite UCI CC BY 4.0 metadata. Manual bounding boxes and per-photo source citations are recorded in manifest.json.")
    result = read(output / "manifest.json")
    # Keep the v4 training selection so this supplement changes only the new source exposure.
    baseline_active = {Path(line).resolve().relative_to(BASE.resolve()).as_posix()
                       for line in (BASE / "train.txt").read_text("utf-8").splitlines() if line}
    baseline_images = {row["image"] for row in base["records"]}
    active_names = baseline_active | {row["image"] for row in result["records"] if row["image"] not in baseline_images}
    (output / "train.txt").write_text("".join((output / row["image"]).as_posix() + "\n"
                                               for row in result["records"] if row["image"] in active_names), encoding="utf-8")
    result["selected_train_images"] = len(active_names)
    result["active_instances_per_split"] = expansion.counts([row for row in result["records"]
                                                            if row["split"] != "train" or row["image"] in active_names])
    result["training_selection"] = "Preserved v4 active train images plus reviewed RealWaste supplement"
    result.update(status="EXPERIMENTAL_NONCOMMERCIAL_RESEARCH", source_license_conflict=True,
                  new_source_near_duplicates_excluded=excluded,
                  benchmark_sha256=expansion.digest(ROOT / "benchmarks/camera-replay-v1.json"))
    result["limitations"] += ["RealWaste research additions retain author CC BY-NC-SA 4.0 terms; resolve UCI/author license discrepancy before broader use.",
                              "All new RealWaste photos share one conservative proxy group on train. Existing holdouts are preserved.",
                              "Manual boxes were screened by Codex; independent human annotation review is pending."]
    expansion.write_json(output / "manifest.json", result)
    expansion.write_json(output / "publisher-tree-snapshot.json", read(metadata))
    expansion.write_json(ROOT / "evaluations/realwaste-v5-build.json", {k: v for k, v in result.items() if k not in {"records", "rejected"}})
    print(json.dumps({"new_images": result["new_images"], "cardboard_train": result["active_instances_per_split"]["train"]["cardboard"],
                      "status": result["status"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "audit", "pixels", "sample", "build"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fpi-archive", type=Path)
    args = parser.parse_args()
    if args.stage == "acquire":
        acquire()
    elif args.stage == "audit":
        audit_fpi(args.fpi_archive)
    elif args.stage == "pixels":
        audit_pixels(args.fpi_archive or CACHE / "fpi-images.zip")
    elif args.stage == "sample":
        export_fpi_sample((args.output or ROOT / "data/local/fpi-det-audit-sample").resolve())
    else:
        build((args.output or OUTPUT).resolve())


if __name__ == "__main__":
    main()
