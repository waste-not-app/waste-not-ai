"""Restore audited data and gate new, independently reviewed YOLO tuning intake.

This command never trains a model or approves annotations on a reviewer's behalf.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

import yaml

from prepare_broad_waste import LABELS

FOCUS = ("cardboard", "plastic_wrapper", "paper", "drink_carton")
TRAIN_TARGET = dict(zip(FOCUS, (150, 100, 100, 100)))
SPLITS = ("train", "val", "test")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, relative):
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes root")
    return resolved


def validate_boxes(boxes):
    for box in boxes:
        if len(box) != 5 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
            raise ValueError("Invalid or nonfinite bbox")
        cls, x, y, w, h = box
        if isinstance(cls, bool) or int(cls) != cls or not 0 <= cls < len(LABELS):
            raise ValueError("Invalid class ID")
        if not (w > 0 and h > 0 and 0 <= x-w/2 < x+w/2 <= 1 and 0 <= y-h/2 < y+h/2 <= 1):
            raise ValueError("Box edges outside image")


def restore(source, output):
    """Rebase split lists without changing source bytes, splits, or active sampling."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Choose a new output separate from the source")
    manifest = read(source / "manifest.json")
    if manifest["labels"] != LABELS:
        raise ValueError("Expected Broad Waste ten-class taxonomy")
    images = {}
    for row in manifest["records"]:
        image = inside(source, row["image"])
        if row["split"] not in SPLITS or image.parent != source / "images" / row["split"]:
            raise ValueError("Image path and split disagree")
        if row["image"] in images or digest(image) != row["prepared_sha256"]:
            raise ValueError("Duplicate image or checksum mismatch")
        label = inside(source, f"labels/{row['split']}/{image.stem}.txt")
        actual = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        if len(actual) != len(row["boxes"]) or any(
            len(a) != 5 or len(b) != 5 or any(not math.isfinite(x) or abs(x-y) > 1e-7 for x, y in zip(a, b))
            for a, b in zip(actual, row["boxes"])
        ):
            raise ValueError("Stored labels differ from manifest")
        images[row["image"]] = (image, label)
    lists = {}
    for name in ("train.txt", "val.txt", "test.txt", "train-full.txt"):
        path = source / name
        if not path.exists() and name == "train-full.txt":
            continue
        entries = []
        for line in path.read_text().splitlines():
            # Legacy lists carry absolute paths from the original monorepo.
            parts = line.replace("\\", "/").rsplit("/images/", 1)
            if len(parts) != 2:
                raise ValueError("Cannot relocate split-list entry")
            relative = "images/" + parts[1]
            if relative not in images or images[relative][0].parent.name != name.split("-")[0].split(".")[0]:
                raise ValueError("Unknown or wrong-split list entry")
            entries.append(relative)
        if len(entries) != len(set(entries)):
            raise ValueError("Duplicate split-list entry")
        expected = {r["image"] for r in manifest["records"] if r["split"] == name.split("-")[0].split(".")[0]}
        if not set(entries) <= expected or (name != "train.txt" and set(entries) != expected):
            raise ValueError("Split-list membership differs from manifest")
        lists[name] = entries
    config = yaml.safe_load((source / "data.yaml").read_text())
    names = config.get("names")
    if isinstance(names, dict):
        names = [names.get(i) for i in range(len(LABELS))]
    if names != LABELS or any(config.get(s) != s + ".txt" for s in SPLITS):
        raise ValueError("Dataset YAML taxonomy or split lists differ")
    output.mkdir(parents=True)
    for relative, (image, label) in images.items():
        target = inside(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, target)
        label_target = output / "labels" / image.parent.name / label.name
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(label, label_target)
    # Preserve provenance bytes, reviews, notices, and balancing metadata.
    for path in source.iterdir():
        if path.is_file() and path.name not in lists and path.name != "data.yaml":
            shutil.copyfile(path, output / path.name)
    for name, entries in lists.items():
        (output / name).write_text("".join((output / entry).as_posix() + "\n" for entry in entries), encoding="utf-8")
    config["path"] = output.as_posix()
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {"source_manifest_sha256": digest(source / "manifest.json"),
              "restored_manifest_sha256": digest(output / "manifest.json"),
              "images": len(images), "split_list_counts": {k: len(v) for k, v in lists.items()},
              "changed": ["data.yaml path", "absolute split-list roots"],
              "new_training_photos": 0, "new_holdout_photos": 0}
    write(output / "relocation.json", report)
    return report


def stage_review(cache, base, output):
    """Recover unused TACO candidates without changing historical review decisions."""
    from PIL import Image, ImageDraw
    cache, output = cache.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Review output already exists")
    used = {(r["source"], str(r["source_id"])) for r in base["records"]}
    decisions = {}
    for name in ("review.json", "review-batch2.json"):
        if (cache / name).exists():
            for row in read(cache / name):
                decisions.setdefault(str(row["source_id"]), []).append(row)
    candidates = []
    for row in read(cache / "acquisition.json")["records"]:
        if (row["source"], str(row["source_id"])) in used:
            continue
        image = inside(cache, f"images/{row['source']}-{row['source_id']}.jpg")
        if digest(image) != row["raw_sha256"]:
            raise ValueError("Candidate checksum mismatch")
        candidates.append((row, image))
    output.mkdir(parents=True)
    records = []
    for row, image in candidates:
        target = output / "images" / image.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(image, target)
        records.append({**row, "cached_path": target.relative_to(output).as_posix(),
                        "prior_reviews": decisions.get(str(row["source_id"]), []),
                        "status": "REVIEW_REQUIRED_NOT_TRAINING_DATA",
                        "holdout_eligible": False})
    for start in range(0, len(records), 12):
        sheet = Image.new("RGB", (1200, 1000), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(records[start:start+12]):
            with Image.open(output / row["cached_path"]) as original:
                thumb = original.convert("RGB")
                thumb.thumbnail((290, 270))
            x, y = offset % 4 * 300, offset // 4 * 330
            sheet.paste(thumb, (x, y))
            for cls, cx, cy, w, h in row["boxes"]:
                draw.rectangle((x+(cx-w/2)*thumb.width, y+(cy-h/2)*thumb.height,
                                x+(cx+w/2)*thumb.width, y+(cy+h/2)*thumb.height), outline="red", width=2)
            draw.text((x, y+275), f"TACO {row['source_id']} - REVIEW REQUIRED", fill="black")
            draw.text((x, y+295), ", ".join(sorted({LABELS[int(b[0])] for b in row["boxes"]})), fill="black")
            thumb.close()
        sheet.save(output / f"contact-{start//12+1:02}.jpg")
        sheet.close()
    write(output / "review-queue.json", {"status": "REVIEW_REQUIRED_NOT_TRAINING_DATA", "records": records})
    summary = {"candidates": len(records), "approved_training_images": 0, "new_holdout_images": 0,
               "candidate_instances": dict(Counter(LABELS[int(b[0])] for r in records for b in r["boxes"])),
               "prior_decisions_preserved": True}
    write(output / "summary.json", summary)
    return summary


def readiness(base, intake, intake_root, protected=()):
    """Return measured gaps; legacy samples never count as new holdout support."""
    if base["labels"] != LABELS or intake.get("labels") != LABELS:
        raise ValueError("Taxonomy differs from Broad Waste")
    protected_rows = list(base["records"]) + list(protected)
    hashes = {r.get("prepared_sha256", r.get("sha256")) for r in protected_rows}
    groups = {r.get("group") for r in protected_rows}
    phashes = [r["phash"] for r in protected_rows if "phash" in r]
    valid, rejected = [], []
    seen_ids, seen_hashes = set(), set()
    for row in intake.get("records", []):
        reasons = []
        try:
            if not row.get("id") or row["id"] in seen_ids:
                raise ValueError("Missing or duplicate ID")
            seen_ids.add(row["id"])
            if row.get("split") not in SPLITS or not row.get("group") or not row.get("session_id"):
                raise ValueError("Missing split, group, or capture session")
            image = inside(intake_root, row["image"])
            checksum = digest(image)
            if checksum != row.get("prepared_sha256"):
                raise ValueError("Image checksum mismatch")
            if checksum in hashes or checksum in seen_hashes or row["group"] in groups:
                raise ValueError("Reused/protected image or group")
            seen_hashes.add(checksum)
            if not isinstance(row.get("phash"), int) or not 0 <= row["phash"] < 2**63:
                raise ValueError("Missing or invalid image pHash")
            # Recompute pixel evidence; never trust the intake's duplicate claims.
            from PIL import Image
            from prepare_drinking_waste import perceptual_hash
            with Image.open(image) as opened:
                if opened.getexif().get(274, 1) != 1:
                    raise ValueError("Normalize orientation and bbox before intake")
                actual_phash = perceptual_hash(opened.convert("RGB"))
            if actual_phash != row["phash"]:
                raise ValueError("Image pHash mismatch")
            if any((row["phash"] ^ value).bit_count() <= 5 for value in phashes):
                raise ValueError("Near-duplicate of protected data")
            validate_boxes(row["boxes"])
            rights = row.get("rights", {})
            if not (rights.get("accepted") is True and rights.get("status") == "RELEASE_ELIGIBLE"
                    and rights.get("license_url") and rights.get("source_page") and rights.get("attribution")):
                raise ValueError("Release rights/attribution unresolved")
            review = row.get("visual_review", {})
            if not (review.get("decision") == "APPROVED" and review.get("reviewer")
                    and review.get("reviewer_type") == "human" and review.get("reviewed_at")
                    and review.get("all_objects_annotated") is True and review.get("labels") == LABELS):
                raise ValueError("Independent complete human review pending")
            if not row["boxes"] and not review.get("verified_negative"):
                raise ValueError("Empty labels lack verified-negative review")
            valid.append(row)
        except (ValueError, KeyError, OSError) as error:
            reasons.append(str(error))
        if reasons:
            rejected.append({"id": row.get("id"), "reasons": reasons})
    # Reject both ends of a cross-split collision; iteration order cannot approve one.
    collisions = set()
    for i, row in enumerate(valid):
        for j, other in enumerate(valid[:i]):
            if row["split"] != other["split"] and (
                row["group"] == other["group"] or row["session_id"] == other["session_id"]
                or (row["phash"] ^ other["phash"]).bit_count() <= 5
            ):
                collisions.update((i, j))
    for index in sorted(collisions):
        rejected.append({"id": valid[index]["id"], "reasons": ["Cross-split group/session/pHash collision"]})
    valid = [r for i, r in enumerate(valid) if i not in collisions]
    # Count connected capture groups, not caller-supplied group aliases.
    parents = list(range(len(valid)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, row in enumerate(valid):
        for j, other in enumerate(valid[:i]):
            if row["group"] == other["group"] or row["session_id"] == other["session_id"]:
                parents[find(i)] = find(j)
    capture_groups = {r["id"]: find(i) for i, r in enumerate(valid)}
    support = {}
    for split in SPLITS:
        rows = [r for r in valid if r["split"] == split]
        support[split] = {}
        for label in FOCUS:
            cls = LABELS.index(label)
            matching = [r for r in rows if any(b[0] == cls for b in r["boxes"])]
            instances = sum(b[0] == cls for r in matching for b in r["boxes"])
            target = TRAIN_TARGET[label] if split == "train" else 50
            count = len(matching) if split == "train" else instances
            independent = len({capture_groups[r["id"]] for r in matching})
            support[split][label] = {"images": len(matching), "instances": instances,
                "groups": independent, "target": target,
                "count_deficit": max(0, target-count),
                "group_deficit": max(0, 20-independent) if split != "train" else 0}
    ready = not rejected and all(not v["count_deficit"] and not v["group_deficit"]
                               for split in support.values() for v in split.values())
    legacy = {split: dict(Counter(LABELS[int(b[0])] for r in base["records"]
                                 if r["split"] == split for b in r["boxes"])) for split in SPLITS}
    return {"status": "READY_FOR_TRAINING_EXPERIMENT" if ready else "INCOMPLETE_DATA_REVIEW",
            "legacy_instances": legacy, "new_support": support, "accepted_new_images": len(valid),
            "rejected": rejected, "holdout_policy": "new test split is final holdout; never select on it",
            "scope": "Data readiness only; does not establish model accuracy"}


def build(base_root, intake_path, protected, output):
    """Merge approved intake, keeping legacy test out of the new final holdout."""
    base_root, output = base_root.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(base_root) or base_root.is_relative_to(output):
        raise ValueError("Choose a new output separate from baseline")
    base, intake = read(base_root / "manifest.json"), read(intake_path)
    report = readiness(base, intake, intake_path.parent, protected)
    if report["status"] != "READY_FOR_TRAINING_EXPERIMENT":
        raise ValueError("Intake data/review quotas incomplete; inspect audit report before build")
    records = []
    files = []
    for row in base["records"]:
        if row["split"] == "test":
            continue
        source = inside(base_root, row["image"])
        if digest(source) != row["prepared_sha256"]:
            raise ValueError("Baseline image checksum mismatch")
        records.append(dict(row))
        files.append((source, row["image"], inside(base_root, f"labels/{row['split']}/{source.stem}.txt")))
    old_active = {Path(line.replace("\\", "/")).name for line in (base_root / "train.txt").read_text().splitlines()}
    for row in intake["records"]:
        source = inside(intake_path.parent, row["image"])
        relative = f"images/{row['split']}/NewCapture-{row['prepared_sha256']}{source.suffix.lower()}"
        review = dict(row["visual_review"])
        if not row["boxes"]:
            review["general_negative"] = True
        records.append({**row, "image": relative, "source": "NewCapture", "source_id": row["id"],
                        "visual_review": review,
                        "kind": "VERIFIED_NEGATIVE" if not row["boxes"] else "ANNOTATED_OBJECTS"})
        files.append((source, relative, None))
    if len({r["image"] for r in records}) != len(records):
        raise ValueError("Output filename collision")
    output.mkdir(parents=True)
    for row, (source, relative, label_source) in zip(records, files):
        target = inside(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        label = output / "labels" / row["split"] / target.with_suffix(".txt").name
        label.parent.mkdir(parents=True, exist_ok=True)
        if label_source:
            shutil.copyfile(label_source, label)
        else:
            label.write_text("".join(" ".join(map(str, b)) + "\n" for b in row["boxes"]), encoding="utf-8")
    active = [r for r in records if r["split"] != "train" or r["source"] == "NewCapture"
              or Path(r["image"]).name in old_active]
    counts = lambda rows: {s: {name: sum(int(b[0]) == i for r in rows if r["split"] == s for b in r["boxes"])
                              for i, name in enumerate(LABELS)} for s in SPLITS}
    manifest = {"version": "broad-waste-issue8-v5", "status": "EXPERIMENTAL_NOT_PROMOTED", "labels": LABELS,
                "records": records, "baseline_manifest_sha256": digest(base_root / "manifest.json"),
                "intake_sha256": digest(intake_path), "instances_per_split": counts(records),
                "active_instances_per_split": counts(active),
                "selected_train_images": sum(r["split"] == "train" for r in active),
                "legacy_test_policy": "Retained in baseline dataset for regression; never mixed into new test holdout"}
    write(output / "manifest.json", manifest)
    for split in SPLITS:
        (output / f"{split}.txt").write_text("".join((output / r["image"]).as_posix()+"\n"
                                                    for r in active if r["split"] == split), encoding="utf-8")
    (output / "train-full.txt").write_text("".join((output / r["image"]).as_posix()+"\n"
                                                    for r in records if r["split"] == "train"), encoding="utf-8")
    (output / "data.yaml").write_text(yaml.safe_dump({"path": output.as_posix(), "names": LABELS,
                                                      **{s: s+".txt" for s in SPLITS}}), encoding="utf-8")
    notices = (base_root / "DATASET_NOTICES.md").read_text(encoding="utf-8")
    notices += "\n\n## New capture attribution\n\n" + "\n".join(
        f"- {r['id']}: {r['rights']['attribution']} — {r['rights']['source_page']} — {r['rights']['license_url']}"
        for r in intake["records"])
    (output / "DATASET_NOTICES.md").write_text(notices+"\n", encoding="utf-8")
    write(output / "readiness.json", report)
    return {"status": manifest["status"], "images": len(records), "new_images": len(intake["records"]),
            "next_check": "Run verify_broad_data.py on the completed output before any training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("restore")
    recover.add_argument("--source", required=True, type=Path)
    recover.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--base", required=True, type=Path)
    audit.add_argument("--intake", required=True, type=Path)
    audit.add_argument("--protected", action="append", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    review = commands.add_parser("stage-review")
    review.add_argument("--cache", required=True, type=Path)
    review.add_argument("--base", required=True, type=Path)
    review.add_argument("--output", required=True, type=Path)
    merge = commands.add_parser("build")
    merge.add_argument("--base", required=True, type=Path)
    merge.add_argument("--intake", required=True, type=Path)
    merge.add_argument("--protected", action="append", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "restore":
        result = restore(args.source, args.output)
    elif args.command == "stage-review":
        result = stage_review(args.cache, read(args.base / "manifest.json"), args.output)
    elif args.command == "build":
        protected = [r for path in args.protected for r in read(path)["records"]]
        result = build(args.base, args.intake, protected, args.output)
    else:
        protected = [r for path in args.protected for r in read(path)["records"]]
        result = readiness(read(args.base / "manifest.json"), read(args.intake), args.intake.parent, protected)
        write(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if args.command == "audit" and result["status"] != "READY_FOR_TRAINING_EXPERIMENT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
