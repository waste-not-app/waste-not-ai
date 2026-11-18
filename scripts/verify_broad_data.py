"""Verify the actual split lists, labels, source rights and image hashes used for training."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/local/broad-waste-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/broad-data-verification.json")
    args = parser.parse_args()
    data = args.data.resolve()
    manifest = json.loads((data / "manifest.json").read_text("utf-8"))
    records = manifest["records"]
    assert len({r["image"] for r in records}) == len(records)
    groups, seen, edge_excursions = {}, {}, []
    for r in records:
        path = data / r["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["prepared_sha256"], path
        previous = seen.setdefault(r["prepared_sha256"], r["split"])
        assert previous == r["split"], "Identical prepared image spans splits"
        assert groups.setdefault(r["group"], r["split"]) == r["split"], "Proxy group spans splits"
        label = data / "labels" / r["split"] / path.with_suffix(".txt").name
        actual = [list(map(float, line.split())) for line in label.read_text("utf-8").splitlines() if line]
        assert len(actual) == len(r["boxes"])
        if not actual:
            assert r.get("kind") == "VERIFIED_NEGATIVE"
            assert r["visual_review"]["decision"] == "APPROVED" and r["visual_review"]["general_negative"]
        for a, b in zip(actual, r["boxes"]):
            assert len(a) == 5 and 0 <= a[0] < len(manifest["labels"]) and int(a[0]) == a[0]
            assert all(abs(x-y) < 1e-7 for x, y in zip(a, b))
            _, x, y, w, h = a
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            excursion = max(0, -(x-w/2), -(y-h/2), x+w/2-1, y+h/2-1)
            # The already-audited original source permits <=.002 boundary rounding.
            tolerance = .002 if r["source"] == "DrinkingWasteV2" else 1e-7
            assert excursion <= tolerance + 1e-8
            if excursion > 1e-7:
                edge_excursions.append({"image": r["image"], "excursion": excursion})
        if r["source"] != "DrinkingWasteV2":
            assert r["rights"]["accepted"] and r["rights"]["license_url"] and r["rights"]["source_page"]
    active_images = {}
    for split in ("train", "val", "test"):
        lines = (data / (split + ".txt")).read_text("utf-8").splitlines()
        listed = set(lines)
        assert len(lines) == len(listed), "Duplicate training/evaluation list entries"
        expected = {(data / r["image"]).as_posix() for r in records if r["split"] == split}
        if split == "train" and "active_instances_per_split" in manifest:
            assert listed <= expected, "Balanced train contains a holdout or unknown image"
            assert len(listed) == manifest["selected_train_images"]
            full = (data / "train-full.txt").read_text("utf-8").splitlines()
            assert len(full) == len(set(full)) and set(full) == expected
        else:
            assert listed == expected
        active_images[split] = len(listed)
        if "active_instances_per_split" in manifest:
            actual = Counter(int(b[0]) for r in records if (data / r["image"]).as_posix() in listed for b in r["boxes"])
            assert {name: actual[i] for i, name in enumerate(manifest["labels"])} == manifest["active_instances_per_split"][split]
    for i, r in enumerate(records):
        for other in records[:i]:
            if other["split"] != r["split"]:
                assert (r["phash"] ^ other["phash"]).bit_count() > 5, "Near-duplicate spans splits"
    output = {"images": len(records), "splits": dict(Counter(r["split"] for r in records)),
              "image_sha256": "all match manifest", "label_geometry": "normalized centers/sizes valid; original DW edge tolerance .002 retained",
              "legacy_edge_rounding": edge_excursions,
              "proxy_group_and_phash_split_overlap": "none", "split_lists": "match manifest", "active_images": active_images,
              "scope": "Data consistency, not independent field accuracy or full annotation completeness"}
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
