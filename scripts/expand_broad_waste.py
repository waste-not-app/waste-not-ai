"""Acquire rare-class TACO examples without modifying the audited baseline dataset."""

import argparse
import hashlib
import json
import random
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from prepare_broad_waste import LABELS, LICENSES, TACO_MAP, normalized_box

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/local/broad-waste-v1"
SOURCE = ROOT / "data/local/downloads/broad-waste-v1"
CACHE = ROOT / "data/local/downloads/broad-waste-v3"
REPORT = ROOT / "evaluations/broad-waste-v3-acquisition.json"
SPLITS = ("train", "val", "test")
RARE = set(range(3, 10))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def counts(records):
    return {
        s: {
            name: sum(int(b[0]) == cls for r in records if r["split"] == s for b in r["boxes"])
            for cls, name in enumerate(LABELS)
        }
        for s in SPLITS
    }


def make_plan(base, taco, target):
    if base["labels"] != LABELS:
        raise ValueError("Baseline taxonomy differs from the expected ten classes")
    old = {str(r["source_id"]) for r in base["records"] if r["source"] == "TACO"}
    # Previously excluded data must never be silently reintroduced as new examples.
    old.update(str(r["source_id"]) for r in base.get("rejected", []) if r["source"] == "TACO")
    annotations = defaultdict(list)
    for ann in taco["annotations"]:
        if ann["category_id"] in TACO_MAP:
            annotations[ann["image_id"]].append(ann)
    pool, rejected = [], []
    for image in taco["images"]:
        anns = annotations[image["id"]]
        if str(image["id"]) in old or not any(TACO_MAP[a["category_id"]] in RARE for a in anns):
            continue
        if any(a.get("iscrowd") for a in anns):
            continue
        filename = Path(image["file_name"])
        number = re.search(r"\d+", filename.stem)
        if not number:
            raise ValueError("TACO filename cannot establish the baseline proxy group")
        try:
            boxes = [
                normalized_box(a["bbox"], image["width"], image["height"], TACO_MAP[a["category_id"]]) for a in anns
            ]
        except ValueError as exc:
            rejected.append({"source_id": str(image["id"]), "reason": str(exc)})
            continue
        pool.append(
            {
                "source": "TACO",
                "source_id": str(image["id"]),
                "url": image.get("flickr_640_url") or image["flickr_url"],
                "original_url": image["flickr_url"],
                "width": image["width"],
                "height": image["height"],
                "group": "TACO-" + filename.parent.name + "-" + str(int(number.group()) // 20),
                "boxes": boxes,
            }
        )
    available = Counter(int(b[0]) for r in pool for b in r["boxes"])
    baseline = counts(base["records"])
    deficits = {name: max(0, target - baseline["train"][name]) for name in LABELS}
    # Rare classes first; retain every mapped target box in every selected scene.
    random.Random(42).shuffle(pool)
    pool.sort(key=lambda r: min(available[int(b[0])] for b in r["boxes"] if int(b[0]) in RARE))
    return {
        "version": "broad-waste-v3",
        "target_train_instances_per_class": target,
        "baseline_counts": baseline,
        "train_deficit": deficits,
        "candidate_images": len(pool),
        "candidate_instances_upper_bound": {name: available[i] for i, name in enumerate(LABELS)},
        "limitations": [
            "Candidate counts are not acquired or approved training examples.",
            "Counts precede rights checks, download failures, visual review, deduplication and splitting.",
            "TACO alone cannot necessarily meet the per-class target; additional sources or own photos are needed.",
            "Do not use classification labels or whole-image boxes as detection ground truth.",
        ],
        "invalid_annotation_images": rejected,
        "candidates": pool,
    }


def read_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "WasteNotDatasetAudit/1.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read(), response.url


def check_rights(url):
    match = re.search(r"/([0-9]+)_", url)
    if not match:
        return {"accepted": False, "reason": "no_flickr_photo_identity"}
    photo_id = match.group(1)
    page = f"https://www.flickr.com/photo.gne?id={photo_id}"
    raw, final = read_url(page)
    if not re.search(r"/photos/[^/]+/" + photo_id + r"(?:/|$)", final):
        return {"accepted": False, "reason": "photo_identity_redirect"}
    license_match = re.search(r'"photoModel"\s*:\s*\{.*?"license"\s*:\s*(\d+)', raw.decode("utf-8"), re.S)
    if not license_match:
        return {"accepted": False, "reason": "missing_image_license"}
    license_id = int(license_match.group(1))
    return {
        "accepted": license_id in LICENSES,
        "license_id": license_id,
        "license_url": LICENSES.get(license_id),
        "source_page": final,
        "creator_page": final.rsplit("/" + photo_id, 1)[0] + "/",
        "verification": "image-specific Flickr page, not toolkit or archive license",
        "checked_at": datetime.now(UTC).isoformat(),
    }


def acquire(plan, cache, max_requests):
    state_path = cache / "acquisition.json"
    state = json.loads(state_path.read_text("utf-8")) if state_path.exists() else {"records": [], "rights": {}}
    for key in ("baseline_manifest_sha256", "source_metadata_sha256"):
        if key in state and state[key] != plan[key]:
            raise ValueError("Source snapshot changed; use a new cache version instead of mixing acquisitions")
        state[key] = plan[key]
    acquired = {r["source_id"] for r in state["records"]}
    cached_rights = {
        str(r["taco_image_id"]): r
        for r in json.loads((SOURCE / "taco-rights.json").read_text("utf-8"))
        if r["accepted"]
    }
    cached_rights.update(state["rights"])
    requests = 0
    state.update(status="RUNNING", last_error=None)
    for row in plan["candidates"]:
        ident = row["source_id"]
        if ident in acquired:
            continue
        right = cached_rights.get(ident)
        if right is not None and not right["accepted"]:
            continue
        if requests >= max_requests:
            state["status"] = "REQUEST_BUDGET_REACHED"
            break
        try:
            if right is None:
                requests += 1
                right = check_rights(row["original_url"])
                state["rights"][ident] = right
            if not right["accepted"]:
                continue
            path = cache / "images" / ("TACO-" + ident + ".jpg")
            if not path.exists():
                if requests >= max_requests:
                    state["status"] = "REQUEST_BUDGET_REACHED"
                    break
                requests += 1
                raw, _ = read_url(row["url"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".part")
                temp.write_bytes(raw)
                # Decode before promoting the cache file; never trust a .jpg suffix.
                with Image.open(temp) as opened:
                    opened.verify()
                temp.replace(path)
            with Image.open(path) as opened:
                if abs(opened.width / opened.height - row["width"] / row["height"]) > 0.01:
                    raise ValueError("Downloaded aspect ratio differs from source annotations")
            state["records"].append(
                {
                    **row,
                    "rights": right,
                    "raw_sha256": digest(path),
                    "cached_path": str(path.resolve()),
                    "visual_review": "PENDING",
                }
            )
            acquired.add(ident)
            if len(acquired) % 10 == 0:
                print(f"Acquired {len(acquired)} images; requests this run: {requests}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # Stop on infrastructure failure/429 instead of hammering the source or bypassing its limits.
            state["status"] = "BLOCKED_NETWORK" if isinstance(exc, urllib.error.URLError) else "BLOCKED_SOURCE"
            state["last_error"] = type(exc).__name__ + ": " + str(exc)
            break
        finally:
            state["requests_this_run"] = requests
            write_json(state_path, state)
    if state["status"] == "RUNNING":
        state["status"] = "COMPLETE"
    state["requests_this_run"] = requests
    write_json(state_path, state)
    return state


def import_browser_downloads(plan, cache, downloads):
    """Import actual browser downloads with the image-specific rights observed in its UI."""
    path = cache / "acquisition.json"
    state = json.loads(path.read_text("utf-8")) if path.exists() else {"records": [], "rights": {}}
    for key in ("baseline_manifest_sha256", "source_metadata_sha256"):
        if key in state and state[key] != plan[key]:
            raise ValueError("Source snapshot changed")
        state[key] = plan[key]
    candidates = {r["source_id"]: r for r in plan["candidates"]}
    known = {r["source_id"]: r for r in state["records"]}
    additions = []
    for evidence in downloads:
        row = candidates[evidence["source_id"]]
        photo_id = re.search(r"/([0-9]+)_", row["original_url"]).group(1)
        if not re.fullmatch(r"https://www\.flickr\.com/photos/[^/]+/" + photo_id + "/", evidence["source_page"]):
            raise ValueError("Browser photo identity differs from the annotations")
        if not re.fullmatch(
            r"https://live\.staticflickr\.com/\d+/" + photo_id + r"_[a-z0-9]+_[a-z]\.jpg", evidence["download_url"]
        ):
            raise ValueError("Browser media URL differs from the photo identity")
        if (
            evidence["license_url"] not in LICENSES.values()
            or not evidence.get("author")
            or not evidence.get("checked_at")
        ):
            raise ValueError("Missing accepted image-specific attribution")
        datetime.fromisoformat(evidence["checked_at"])
        source = Path(evidence["local_path"])
        image_hash = digest(source)
        with Image.open(source) as opened:
            if abs(opened.width / opened.height - row["width"] / row["height"]) > 0.01:
                raise ValueError("Browser image aspect ratio differs from source annotations")
            opened.verify()
        if row["source_id"] in known:
            if known[row["source_id"]]["raw_sha256"] != image_hash:
                raise ValueError("A different browser image is already cached for this source")
            continue
        dest = cache / "images" / ("TACO-" + row["source_id"] + ".jpg")
        if dest.exists() and digest(dest) != image_hash:
            raise ValueError("Refusing to overwrite a different cached image")
        right = {k: evidence[k] for k in ("source_page", "creator_page", "license_url", "author", "checked_at")}
        right.update(accepted=True, verification="Flickr image license and attribution observed in browser DOM")
        additions.append(
            (
                {
                    **row,
                    "url": evidence["download_url"],
                    "rights": right,
                    "raw_sha256": image_hash,
                    "cached_path": str(dest.resolve()),
                    "visual_review": "PENDING",
                },
                source,
                dest,
            )
        )
        known[row["source_id"]] = {"raw_sha256": image_hash}
    # Validate the entire batch before changing the acquisition checkpoint.
    for row, source, dest in additions:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(source, dest)
        state["records"].append(row)
        state["rights"][row["source_id"]] = row["rights"]
    state.update(
        status="PARTIALLY_ACQUIRED",
        last_error=None,
        requests_this_run=0,
        browser_images_imported_last_batch=len(additions),
    )
    write_json(path, state)
    return state


def split_new_records(base_records, new_records):
    """Lock all old splits and reject new components which bridge old holdouts."""
    rows = [*base_records, *new_records]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    groups, hashes = {}, {}
    for i, row in enumerate(rows):
        if row["group"] in groups:
            union(i, groups[row["group"]])
        groups[row["group"]] = i
        for key in ("raw_sha256", "prepared_sha256"):
            if row.get(key):
                if row[key] in hashes:
                    union(i, hashes[row[key]])
                hashes[row[key]] = i
        for j in range(i):
            if (int(row["phash"]) ^ int(rows[j]["phash"])).bit_count() <= 5:
                union(i, j)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    assigned, rejected = [], []
    current = {s: Counter(int(b[0]) for r in base_records if r["split"] == s for b in r["boxes"]) for s in SPLITS}
    totals = Counter(int(b[0]) for r in rows for b in r["boxes"])
    seen = {r.get(k) for r in base_records for k in ("raw_sha256", "prepared_sha256") if r.get(k)}
    for indices in components.values():
        additions = [i for i in indices if i >= len(base_records)]
        locked = {rows[i]["split"] for i in indices if i < len(base_records)}
        if len(locked) > 1:
            rejected.extend({"source_id": rows[i]["source_id"], "reason": "bridges_locked_splits"} for i in additions)
            continue
        component_counts = Counter(int(b[0]) for i in additions for b in rows[i]["boxes"])
        if not component_counts:
            continue
        rarest = min(component_counts, key=lambda cls: totals[cls])
        split = (
            next(iter(locked))
            if locked
            else min(SPLITS, key=lambda s: current[s][rarest] / {"train": 0.7, "val": 0.15, "test": 0.15}[s])
        )
        for i in additions:
            row = rows[i]
            if row["raw_sha256"] in seen:
                rejected.append({"source_id": row["source_id"], "reason": "exact_duplicate"})
                continue
            seen.add(row["raw_sha256"])
            assigned.append({**row, "split": split})
            current[split].update(int(b[0]) for b in row["boxes"])
    return assigned, rejected


def balanced_train(records, cap):
    """Select whole images once, prioritizing rare classes; do not erase other boxes."""
    training = [r for r in records if r["split"] == "train"]
    totals = Counter(int(b[0]) for r in training for b in r["boxes"])
    random.Random(42).shuffle(training)
    training.sort(key=lambda r: min(totals[int(b[0])] for b in r["boxes"]))
    selected, exposure = [], Counter()
    for row in training:
        if any(exposure[int(b[0])] < cap for b in row["boxes"]):
            selected.append(row)
            exposure.update(int(b[0]) for b in row["boxes"])
    return selected


def build_dataset(
    base,
    state,
    review,
    output,
    cap,
    *,
    base_dir=None,
    version="broad-waste-v3",
    source_metadata_path=None,
    source_notice=None,
):
    import yaml
    from prepare_drinking_waste import parse_boxes, perceptual_hash

    base_dir = base_dir or BASE
    source_metadata_path = source_metadata_path or SOURCE / "taco.json"
    if state.get("baseline_manifest_sha256") != digest(base_dir / "manifest.json"):
        raise ValueError("Baseline snapshot differs from the acquisition")
    if state.get("source_metadata_sha256") != digest(source_metadata_path):
        raise ValueError("Source annotations differ from the acquisition")
    if output.exists() or output.with_name(output.name + ".building").exists():
        raise ValueError("Output or staging directory exists; preserve it and choose another --output")
    reviews = {r["source_id"]: r for r in review}
    if len(reviews) != len(review):
        raise ValueError("Duplicate image identities in visual review")
    additions = []
    for row in state["records"]:
        checked = reviews.get(row["source_id"], {})
        if checked.get("decision") != "APPROVED":
            continue
        if not checked.get("reviewer", "").strip() or not checked.get("reviewed_at"):
            raise ValueError("An approved image needs an actual reviewer and review date")
        datetime.fromisoformat(checked["reviewed_at"])
        source = Path(row["cached_path"])
        if checked.get("raw_sha256") != row["raw_sha256"] or digest(source) != row["raw_sha256"]:
            raise ValueError("Review/image checksum mismatch")
        if not row["rights"].get("accepted") or not row["rights"].get("license_url"):
            raise ValueError("Missing accepted image-specific rights")
        # Validate the real boxes; never infer a detection box from an image-level class.
        parse_boxes("\n".join(" ".join(map(str, b)) for b in row["boxes"]), len(LABELS))
        with Image.open(source) as opened:
            phash = perceptual_hash(opened.convert("RGB"))
        additions.append({**row, "phash": phash, "visual_review": checked})
    if not additions:
        raise ValueError("No new downloaded, licensed and visually approved images; dataset not created")
    assigned, rejected = split_new_records(base["records"], additions)
    if not assigned:
        raise ValueError("Every new image was rejected as a duplicate or split conflict")
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True)
    records = []
    for row in [*base["records"], *assigned]:
        relative = row.get("image") or f"images/{row['split']}/{row['source']}-{row['source_id']}.jpg"
        dest = staging / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        inherited = "image" in row
        source = base_dir / row["image"] if inherited else Path(row["cached_path"])
        if inherited and digest(source) != row["prepared_sha256"]:
            raise ValueError("Baseline image changed since its audit")
        shutil.copy2(source, dest)
        label = staging / "labels" / row["split"] / (dest.stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "".join(f"{int(b[0])} " + " ".join(f"{v:.8f}" for v in b[1:]) + "\n" for b in row["boxes"]),
            encoding="utf-8",
        )
        records.append({**row, "image": relative, "prepared_sha256": digest(dest)})
    selected = balanced_train(records, cap)
    active = selected + [r for r in records if r["split"] != "train"]
    for split in SPLITS:
        (staging / (split + ".txt")).write_text(
            "".join((output / r["image"]).as_posix() + "\n" for r in active if r["split"] == split), encoding="utf-8"
        )
    (staging / "train-full.txt").write_text(
        "".join((output / r["image"]).as_posix() + "\n" for r in records if r["split"] == "train"), encoding="utf-8"
    )
    (staging / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": output.as_posix(),
                "train": "train.txt",
                "val": "val.txt",
                "test": "test.txt",
                "names": dict(enumerate(LABELS)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": version,
        "labels": LABELS,
        "seed": 42,
        "status": "EXPERIMENTAL",
        "records": records,
        "rejected": rejected,
        "baseline_manifest_sha256": digest(base_dir / "manifest.json"),
        "instances_per_split": counts(records),
        "active_instances_per_split": counts(active),
        "train_instance_cap_per_class": cap,
        "new_images": len(assigned),
        "selected_train_images": len(selected),
        "limitations": [
            "Whole-image selection: a co-occurring class can exceed the instance cap.",
            "Selection does not create new images or guarantee the target count.",
            "All old split assignments are preserved; test was inspected in previous experiments.",
            "TACO filename groups and pHash are proxies, not independently verified physical-object identities.",
        ],
    }
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "acquisition-snapshot.json", state)
    write_json(staging / "review-snapshot.json", review)
    source_notice = (
        source_notice
        or "Additional TACO images: Pedro F. Proença and Pedro Simões; image-specific rights, original URLs and reviewer records are in manifest.json."
    )
    (staging / "DATASET_NOTICES.md").write_text(
        (base_dir / "DATASET_NOTICES.md").read_text("utf-8") + "\n\n" + source_notice + "\n", encoding="utf-8"
    )
    staging.rename(output)
    return {k: v for k, v in manifest.items() if k not in {"records", "rejected"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "acquire", "import-browser", "build"), default="plan")
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--review", type=Path, help="JSON array of image-specific visual review decisions")
    parser.add_argument(
        "--browser-import", type=Path, help="Actual browser downloads and observed image-specific rights"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/local/broad-waste-v3")
    parser.add_argument("--train-cap", type=int, default=600, help="Whole-image majority-class instance cap")
    args = parser.parse_args()
    if args.target < 1 or args.max_requests < 1 or args.train_cap < 1:
        parser.error("target, max-requests and train-cap must be positive")
    base = json.loads((BASE / "manifest.json").read_text("utf-8"))
    if args.stage == "build":
        if not args.review:
            parser.error("build needs --review; unreviewed downloads cannot enter training")
        result = build_dataset(
            base,
            json.loads((CACHE / "acquisition.json").read_text("utf-8")),
            json.loads(args.review.read_text("utf-8")),
            args.output.resolve(),
            args.train_cap,
        )
        result.update(
            prepared_manifest_sha256=digest(args.output / "manifest.json"),
            review_sha256=digest(args.review),
            acquisition_sha256=digest(CACHE / "acquisition.json"),
            acquisition_snapshot_sha256=digest(args.output / "acquisition-snapshot.json"),
            review_snapshot_sha256=digest(args.output / "review-snapshot.json"),
        )
        result["dataset"] = args.output.name
        write_json(ROOT / "evaluations" / (args.output.name + "-build.json"), result)
        print(json.dumps(result, indent=2))
        return 0
    taco = json.loads((SOURCE / "taco.json").read_text("utf-8"))
    plan = make_plan(base, taco, args.target)
    plan["baseline_manifest_sha256"] = digest(BASE / "manifest.json")
    plan["source_metadata_sha256"] = digest(SOURCE / "taco.json")
    CACHE.mkdir(parents=True, exist_ok=True)
    write_json(CACHE / "plan.json", plan)
    if args.stage == "acquire":
        state = acquire(plan, CACHE, args.max_requests)
    elif args.stage == "import-browser":
        if not args.browser_import:
            parser.error("import-browser needs --browser-import")
        state = import_browser_downloads(plan, CACHE, json.loads(args.browser_import.read_text("utf-8")))
    else:
        state_path = CACHE / "acquisition.json"
        state = json.loads(state_path.read_text("utf-8")) if state_path.exists() else None
    report = {k: v for k, v in plan.items() if k != "candidates"}
    prepared_path = args.output / "manifest.json"
    prepared = json.loads(prepared_path.read_text("utf-8")) if prepared_path.exists() else None
    if prepared and prepared.get("baseline_manifest_sha256") != plan["baseline_manifest_sha256"]:
        prepared = None
    report.update(
        status=state["status"] if state else "PLANNED",
        new_images_acquired=len(state["records"]) if state else 0,
        ready_for_training=prepared is not None,
        baseline_modified=False,
        prepared_dataset=args.output.name if prepared else None,
        training_scope="EXPERIMENTAL_ONLY" if prepared else None,
        new_images_in_dataset=prepared["new_images"] if prepared else 0,
        training_target_met=bool(
            prepared and all(v >= args.target for v in prepared["active_instances_per_split"]["train"].values())
        ),
    )
    if state:
        report.update(last_error=state["last_error"], requests_this_run=state["requests_this_run"])
    write_json(REPORT, report)
    print(json.dumps(report, indent=2))
    return 2 if args.stage == "acquire" and state["status"].startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
