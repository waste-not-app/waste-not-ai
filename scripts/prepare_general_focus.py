"""Prepare phone/remote review queues without treating native FPI faces as COCO people."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import httpx
from PIL import Image

from prepare_external_focus import FPI_ARCHIVE_SHA256, parse_fpi_labels, phash_neighbors
import prepare_focus_datasets as focus_data
from prepare_tuning_data import inside, read, write, readiness, restore, build, validate_boxes
from wastenot_ai.taxonomy import COCO_LABELS

ROOT = Path(__file__).resolve().parents[1]
LABELS = list(COCO_LABELS)
FOCUS = ("cell phone", "remote")
TARGETS = {"cell phone": 200, "remote": 150}
FPI_IMAGE_COUNT = 22879
VIEW_TAGS = {"phone_front", "phone_rear", "phone_side", "handheld", "occluded", "remote_short",
             "remote_long", "counterexample_calculator", "counterexample_powerbank", "low_light",
             "small_object", "control_other"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(row):
    return row["source_split"], row["source_id"]


def inspect_fpi(records, pixels, protected):
    """Conservatively quarantine duplicate components, including annotation conflicts."""
    annotations = {key(r): r for r in records}
    by_key = {key(r): r for r in pixels}
    if len(annotations) != len(records) or len(by_key) != len(pixels) or annotations.keys() != by_key.keys():
        raise ValueError("FPI index/pixels have duplicate or mismatched identities")
    reasons = defaultdict(set)
    hash_groups = defaultdict(list)
    for row in pixels:
        hash_groups[row["sha256"]].append(key(row))
    for group in hash_groups.values():
        signatures = {tuple(sorted(tuple(b) for b in annotations[k]["boxes"])) for k in group}
        if len(signatures) > 1:
            for ident in group:
                reasons[ident].add("conflicting_annotations_for_identical_pixels")
    combined = [dict(r, protected=True) for r in protected] + pixels
    for i, j in phash_neighbors(combined):
        a, b = combined[i], combined[j]
        if a.get("protected"):
            continue
        if b.get("protected"):
            reasons[key(a)].add("protected_dataset_near_duplicate")
        else:
            reason = "cross_publisher_split_duplicate" if a["source_split"] != b["source_split"] else "within_split_duplicate"
            reasons[key(a)].add(reason)
            reasons[key(b)].add(reason)
    for row in records:
        try:
            validate_boxes(row["boxes"], 2)
        except ValueError:
            reasons[key(row)].add("invalid_native_geometry")
        if not any(b[0] == 0 for b in row["boxes"]):
            reasons[key(row)].add("no_phone_annotation")
    return reasons


def stage_fpi(archive_path, pixel_path, pixel_report, protected, output, limit=400, prior_reviews=()):
    output = output.resolve()
    if output.exists() or limit < 1:
        raise ValueError("Need a positive limit and a fresh output")
    if digest(archive_path) != FPI_ARCHIVE_SHA256:
        raise ValueError("FPI archive checksum mismatch")
    if digest(pixel_path) != pixel_report["pixel_cache_sha256"]:
        raise ValueError("FPI pixel-cache checksum mismatch")
    cache = read(pixel_path)
    if cache["archive_sha256"] != FPI_ARCHIVE_SHA256:
        raise ValueError("FPI pixel cache belongs to another archive")
    records, combined_lines = [], 0
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        if len(members) != len(archive.namelist()):
            raise ValueError("Duplicate archive member names")
        for name in sorted(members):
            if not name.endswith(".txt"):
                continue
            parts = Path(name).parts
            if len(parts) != 4 or parts[0] != "reorganized_dataset" or parts[1] not in {"train", "val", "test"} or parts[2] != "labels":
                raise ValueError("Unexpected FPI label member layout")
            split, ident = parts[1], Path(name).stem
            image_member = f"reorganized_dataset/{split}/images/{ident}.jpg"
            if image_member not in members:
                raise ValueError("FPI image missing")
            boxes, count = parse_fpi_labels(archive.read(name).decode("utf-8-sig"))
            combined_lines += count
            records.append({"source_id": ident, "source_split": split, "boxes": boxes,
                            "image_member": image_member})
        pixels = cache["records"]
        if len(pixels) != FPI_IMAGE_COUNT:
            raise ValueError("Need the complete audited 22,879-image pixel cache")
        rejected = inspect_fpi(records, pixels, protected)
        prior = {(r["source_split"], r["source_id"]): r for r in prior_reviews}
        if len(prior) != len(prior_reviews):
            raise ValueError("Duplicate historical review identity")
        by_key = {key(r): r for r in pixels}
        for ident, review in prior.items():
            if ident not in by_key or review["raw_sha256"] != by_key[ident]["sha256"]:
                raise ValueError("Historical review refers to different pixels")
        pool = [r for r in records if not rejected.get(key(r)) and r["source_split"] == "train"
                and max(b[3]*b[4] for b in r["boxes"] if b[0] == 0) >= .03]
        pool.sort(key=lambda r: ("rear_visible" not in prior.get(key(r), {}).get("view_tags", []),
                                 hashlib.sha256(("issue9-fpi-v1"+r["source_id"]).encode()).hexdigest()))
        output.mkdir(parents=True)
        selected = []
        from prepare_drinking_waste import perceptual_hash
        for row in pool[:limit]:
            pixel = by_key[key(row)]
            raw = archive.read(row["image_member"])
            if hashlib.sha256(raw).hexdigest() != pixel["sha256"]:
                raise ValueError("FPI image differs from pinned pixel cache")
            with Image.open(BytesIO(raw)) as image:
                if image.size != (pixel["width"], pixel["height"]) or image.getexif().get(274, 1) != 1:
                    raise ValueError("FPI dimensions/orientation mismatch")
                if perceptual_hash(image.convert("RGB")) != pixel["phash"]:
                    raise ValueError("FPI image pHash mismatch")
            target = inside(output, "images/" + row["source_id"] + ".jpg")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(raw)
            label = output / "native-labels" / (row["source_id"]+".txt")
            label.parent.mkdir(exist_ok=True)
            label.write_text("".join(" ".join(map(str, b))+"\n" for b in row["boxes"]), encoding="utf-8")
            selected.append({**row, **pixel, "image": target.relative_to(output).as_posix(),
                             "status": "FULL_COCO_REANNOTATION_AND_REVIEW_REQUIRED",
                             "holdout_eligible": False, "view_tags": [], "grouping_review": "PENDING",
                             "prior_review": prior.get(key(row))})
    manifest = {"version": "issue9-fpi-review-v1", "status": "AUDIT_ONLY_NATIVE_PHONE_FACE",
                "labels": ["phone", "face"], "archive_sha256": FPI_ARCHIVE_SHA256,
                "pixel_cache_sha256": digest(pixel_path), "records": selected,
                "usage": "REVIEW_ONLY_NO_TRAINING", "full_coco_reannotation_required": True}
    write(output / "manifest.json", manifest)
    focus_data.make_sheets([dict(r, cached_path=str(output/r["image"])) for r in selected],
                           output / "sheets", ["phone", "face"], "native")
    summary = {"publisher_images": len(records), "combined_label_lines": combined_lines,
               "quarantined_images": len(rejected), "quarantine_reasons": dict(Counter(reason for v in rejected.values() for reason in v)),
               "candidate_images": len(selected), "approved_training_images": 0, "new_holdout_images": 0,
               "historically_screened_rear_candidates": sum("rear_visible" in (r.get("prior_review") or {}).get("view_tags", []) for r in selected),
               "native_face_to_coco_person_mapping": "FORBIDDEN", "status": manifest["status"]}
    write(output / "summary.json", summary)
    write(output / "quarantine.json", [{"source_split": k[0], "source_id": k[1], "reasons": sorted(v)}
                                       for k, v in sorted(rejected.items())])
    (output / "DATASET_NOTICES.md").write_text(
        "# FPI-Det review queue\n\nPublisher: https://github.com/KvCgRv/FPI-Det\n"
        "Pinned source: 83c452d1a1e5b1bac88afe1a6834cb9c25e036eb; publisher-declared MIT.\n"
        "Phone/face native annotations only. No training YAML; full COCO annotation, capture grouping, rights and human review remain pending.\n",
        encoding="utf-8")
    return summary


def plan_coco(annotations, protected, output, phone=80, remote=180):
    if output.exists() or min(phone, remote) < 0:
        raise ValueError("Need fresh output and nonnegative quotas")
    excluded = {int(r.get("original_source_id", r.get("source_id"))) for r in protected
                if str(r.get("source", "")).startswith("COCO")}
    with zipfile.ZipFile(annotations) as archive:
        data = json.loads(archive.read("annotations/instances_train2017.json"))
    candidates = focus_data.coco_candidates(data, "train2017", excluded)
    selected = focus_data.select_candidates(candidates, [("cell phone", phone), ("remote", remote)])
    result = {"version": "issue9-coco-candidates-v1", "labels": LABELS,
              "usage": "REVIEW_ONLY_NO_TRAINING", "annotation_archive_sha256": digest(annotations),
              "selected": selected, "holdout_eligible": False,
              "limitation": "COCO pretraining/source exposure; these are training candidates, never new final holdout"}
    write(output, result)
    return {"planned_images": len(selected), "target_strata": dict(Counter(r["selection_stratum"] for r in selected))}


def acquire_coco(plan, protected, output, resume=False):
    output = output.resolve()
    if (output.exists() and not resume) or plan.get("labels") != LABELS or plan.get("usage") != "REVIEW_ONLY_NO_TRAINING":
        raise ValueError("Need a registered plan and fresh output")
    from prepare_drinking_waste import perceptual_hash
    blocked = [r["phash"] for r in protected]
    protected_sha = {r.get("prepared_sha256", r.get("sha256")) for r in protected}
    selected, rejected, seen = [], [], []
    if output.exists():
        previous = read(output / "manifest.json")
        expected = {r["source_id"]: r for r in plan["selected"]}
        if previous.get("labels") != LABELS or previous["annotation_archive_sha256"] != plan["annotation_archive_sha256"]:
            raise ValueError("Resume cache belongs to another taxonomy/source snapshot")
        for row in previous["records"]:
            if row["source_id"] not in expected or row["boxes"] != expected[row["source_id"]]["boxes"]:
                raise ValueError("Resume cache labels changed")
            if digest(inside(output, row["image"])) != row["raw_sha256"]:
                raise ValueError("Resume cache image checksum mismatch")
            selected.append(row)
            seen.append(row["phash"])
    else:
        output.mkdir(parents=True)
        (output / "images").mkdir()

    def fetch(row):
        try:
            ident = str(int(row["source_id"]))
            url = f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{int(ident):012d}.jpg"
            validate_boxes(row["boxes"], 80)
            with httpx.Client(timeout=20, follow_redirects=False) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > 16*1024*1024:
                            raise ValueError("Image exceeds research download cap")
            checksum = hashlib.sha256(raw).hexdigest()
            with Image.open(BytesIO(raw)) as image:
                if image.size != (row["width"], row["height"]) or image.getexif().get(274, 1) != 1:
                    raise ValueError("COCO stored pixels disagree with annotations")
                phash = perceptual_hash(image.convert("RGB"))
            if checksum in protected_sha or any((phash ^ v).bit_count() <= 5 for v in blocked):
                raise ValueError("Protected-data duplicate")
            return {**row, "raw_sha256": checksum, "phash": phash, "raw": raw}
        except (httpx.HTTPError, ValueError, OSError) as error:
            return {"source_id": row.get("source_id"), "error": str(error)}

    finished = {r["source_id"] for r in selected}
    if len(finished) != len(selected):
        raise ValueError("Duplicate image identity in resume cache")
    pending = [r for r in plan["selected"] if r["source_id"] not in finished]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(fetch, pending), 1):
            if "error" in row:
                rejected.append(row)
            elif any((row["phash"] ^ v).bit_count() <= 5 for v in seen):
                rejected.append({"source_id": row["source_id"], "error": "Duplicate selected candidate"})
            else:
                raw = row.pop("raw")
                target = inside(output, f"images/COCO-{int(row['source_id'])}.jpg")
                target.write_bytes(raw)
                seen.append(row["phash"])
                selected.append({**row, "image": target.relative_to(output).as_posix(),
                                 "status": "HUMAN_REVIEW_REQUIRED", "holdout_eligible": False,
                                 "annotation_scope": "PUBLISHER_COCO80", "view_tags": []})
            write(output / "manifest.json", {"labels": LABELS, "usage": "REVIEW_ONLY_NO_TRAINING",
                                             "records": selected, "rejected": rejected,
                                             "annotation_archive_sha256": plan["annotation_archive_sha256"]})
            if index % 20 == 0:
                print(f"COCO acquisition {index}/{len(plan['selected'])}; candidates {len(selected)}", flush=True)
    focus_data.make_sheets([dict(r, cached_path=str(output/r["image"])) for r in selected], output/"sheets", LABELS, "coco")
    summary = {"planned_images": len(plan["selected"]), "candidate_images": len(selected),
               "rejected_images": len(rejected), "approved_training_images": 0, "new_holdout_images": 0,
               "target_instances": dict(Counter(LABELS[int(b[0])] for r in selected for b in r["boxes"] if LABELS[int(b[0])] in FOCUS))}
    write(output / "summary.json", summary)
    return summary


def audit_intake(base, intake, root, protected):
    extra = []
    safe = []
    for row in intake.get("records", []):
        if row.get("annotation_scope") != "COCO80_COMPLETE":
            extra.append({"id": row.get("id"), "reasons": ["Full COCO80 reannotation/review required; native FPI labels are not training labels"]})
        elif row.get("source") == "FPI-Det" and row.get("reannotated_from_native") is not True:
            extra.append({"id": row.get("id"), "reasons": ["FPI native phone/face conversion is not complete COCO reannotation"]})
        elif (not isinstance(row.get("view_tags"), list) or not row["view_tags"]
              or any(not isinstance(tag, str) or tag not in VIEW_TAGS for tag in row["view_tags"])
              or len(set(row["view_tags"])) != len(row["view_tags"])):
            extra.append({"id": row.get("id"), "reasons": ["Reviewed view/condition tags missing"]})
        else:
            safe.append(row)
    result = readiness(base, {**intake, "records": safe}, root, protected, labels=LABELS, focus=FOCUS, train_targets=TARGETS)
    result["rejected"].extend(extra)
    if extra:
        result["status"] = "INCOMPLETE_DATA_REVIEW"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("restore")
    recover.add_argument("--source", required=True, type=Path)
    recover.add_argument("--output", required=True, type=Path)
    fpi = commands.add_parser("fpi-queue")
    fpi.add_argument("--archive", required=True, type=Path)
    fpi.add_argument("--pixels", required=True, type=Path)
    fpi.add_argument("--pixel-report", type=Path, default=ROOT/"evaluations/fpi-det-pixel-audit.json")
    fpi.add_argument("--limit", type=int, default=400)
    fpi.add_argument("--prior-reviews", type=Path, default=ROOT/"annotations/fpi-det-sample-v1.json")
    plan = commands.add_parser("coco-plan")
    plan.add_argument("--annotations", required=True, type=Path)
    plan.add_argument("--phone", type=int, default=80)
    plan.add_argument("--remote", type=int, default=180)
    acquire = commands.add_parser("coco-acquire")
    acquire.add_argument("--plan", required=True, type=Path)
    acquire.add_argument("--resume", action="store_true")
    for name in ("audit", "build"):
        item = commands.add_parser(name)
        item.add_argument("--base", required=True, type=Path)
        item.add_argument("--intake", required=True, type=Path)
    for item in (fpi, plan, acquire, commands.choices["audit"], commands.choices["build"]):
        item.add_argument("--protected", action="append", required=True, type=Path)
        item.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protected = [r for path in getattr(args, "protected", []) for r in read(path)["records"]]
    if args.command == "restore":
        result = restore(args.source, args.output, labels=LABELS)
    elif args.command == "fpi-queue":
        result = stage_fpi(args.archive, args.pixels, read(args.pixel_report), protected, args.output, args.limit,
                           read(args.prior_reviews))
    elif args.command == "coco-plan":
        result = plan_coco(args.annotations, protected, args.output, args.phone, args.remote)
    elif args.command == "coco-acquire":
        result = acquire_coco(read(args.plan), protected, args.output, args.resume)
    else:
        result = audit_intake(read(args.base/"manifest.json"), read(args.intake), args.intake.parent, protected)
        if args.command == "audit":
            write(args.output, result)
        elif result["status"] != "READY_FOR_TRAINING_EXPERIMENT":
            raise ValueError("General intake review/quotas incomplete; inspect audit report")
        else:
            result = build(args.base, args.intake, protected, args.output, labels=LABELS, focus=FOCUS, train_targets=TARGETS)
    print(json.dumps(result))
    if args.command == "audit" and result["status"] != "READY_FOR_TRAINING_EXPERIMENT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
