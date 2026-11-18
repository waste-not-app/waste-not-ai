import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def external_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("external_parser", scripts / "prepare_external_focus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fpi_parser_preserves_all_boxes_on_combined_lines_and_does_not_clip(external_module):
    boxes, combined = external_module.parse_fpi_labels("0 .5 .5 .2 .2 0 .7 .7 .1 .1\n1 -.1 .4 .2 .2\n")
    assert len(boxes) == 3 and combined == 1
    assert boxes[1] == [0, .7, .7, .1, .1]
    assert boxes[2][1] == -.1
    for invalid in ("0 .1 .2", "0 nan .5 .2 .2", "2 .5 .5 .2 .2"):
        with pytest.raises(ValueError):
            external_module.parse_fpi_labels(invalid)


def test_phash_bands_find_all_close_pairs_including_changes_across_five_bands(external_module):
    values = [0, 1, sum(1 << bit for bit in (0, 11, 22, 33, 43)),
              sum(1 << bit for bit in (0, 11, 22, 33, 43, 53)), (1 << 63) - 1,
              ((1 << 63) - 1) ^ (1 << 62)]
    rows = [{"phash": value} for value in values]
    expected = {(i, j) for i, value in enumerate(values) for j in range(i)
                if (value ^ values[j]).bit_count() <= 5}
    assert set(external_module.phash_neighbors(rows)) == expected


def test_fpi_export_keeps_native_face_scope_and_refuses_to_overwrite(external_module, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    image = cache / "source.jpg"
    Image.new("RGB", (100, 100), "blue").save(image)
    checksum = external_module.expansion.digest(image)
    row = {"source_id": "photo", "source_split": "train", "cached_path": str(image),
           "raw_sha256": checksum, "boxes": [[0, .5, .5, .2, .2], [1, .5, .3, .2, .2]]}
    (cache / "fpi-sample.json").write_text(json.dumps({"records": [row]}))
    (cache / "fpi-pixels.json").write_text(json.dumps({"records": [{"source_id": "photo", "phash": 0,
                                                                  "width": 100, "height": 100}]}))
    review = tmp_path / "annotations/fpi-det-sample-v1.json"
    review.parent.mkdir(parents=True)
    review.write_text(json.dumps([{"source_id": "photo", "raw_sha256": checksum, "decision": "PHONE_SCOPE_SCREENED"}]))
    monkeypatch.setattr(external_module, "ROOT", tmp_path)
    monkeypatch.setattr(external_module, "CACHE", cache)
    output = tmp_path / "sample"
    external_module.export_fpi_sample(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "AUDIT_ONLY_NATIVE_PHONE_FACE"
    assert manifest["labels"] == ["phone", "face"]
    assert not (output / "data.yaml").exists()
    assert len((output / "labels/train/photo.txt").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="output exists"):
        external_module.export_fpi_sample(output)


def test_external_build_preserves_active_train_and_pinned_source_rights(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("external_focus", scripts / "prepare_external_focus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import prepare_drinking_waste

    base, cache, output = tmp_path / "base", tmp_path / "cache", tmp_path / "output"
    base.mkdir()
    cache.mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "CACHE", cache)
    monkeypatch.setattr(module, "REVIEW", cache / "review.json")
    monkeypatch.setattr(module, "perceptual_hash", lambda _: 2**50 - 1)
    monkeypatch.setattr(prepare_drinking_waste, "perceptual_hash", lambda _: 2**50 - 1)
    records = []
    for index in range(2):
        relative = f"images/train/old-{index}.jpg"
        image = base / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), (index * 100, 0, 0)).save(image)
        records.append({"source": "OldSource", "source_id": str(index), "image": relative, "split": "train",
                        "prepared_sha256": module.expansion.digest(image), "raw_sha256": module.expansion.digest(image),
                        "group": "old-train", "phash": index, "boxes": [[0, .5, .5, .8, .8]]})
    (base / "manifest.json").write_text(json.dumps({"records": records}))
    (base / "DATASET_NOTICES.md").write_text("Inherited attribution")
    (base / "train.txt").write_text((base / records[0]["image"]).as_posix() + "\n")
    for relative in ("benchmarks/camera-replay-v1.json", "data/local/general-confusion-v1/manifest.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"records": []}')

    def blob(path, publisher_path):
        raw = path.read_bytes()
        return {"path": publisher_path, "type": "blob",
                "sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}

    readme = cache / "realwaste-README.md"
    readme.write_text("Author license CC BY-NC-SA 4.0")
    image = cache / "Cardboard_1.jpg"
    Image.new("RGB", (100, 100), "blue").save(image)
    source = {**blob(image, "RealWaste/Cardboard/Cardboard_1.jpg"), "cached_path": str(image),
              "source_id": "Cardboard_1", "raw_sha256": module.expansion.digest(image)}
    (cache / "realwaste-tree.json").write_text(json.dumps({"sha": module.REALWASTE_COMMIT,
        "tree": [blob(readme, "README.md"), source]}))
    (cache / "realwaste-acquisition.json").write_text(json.dumps({"commit": module.REALWASTE_COMMIT, "selected": [source]}))
    (cache / "review.json").write_text(json.dumps([{"source_id": "Cardboard_1", "decision": "APPROVED",
        "raw_sha256": source["raw_sha256"], "pixel_bbox_xyxy": [10, 10, 90, 90],
        "reviewer": "synthetic-test-fixture", "reviewed_at": "2026-11-07"}]))
    module.build(output)
    manifest = json.loads((output / "manifest.json").read_text())
    listed = (output / "train.txt").read_text().splitlines()
    assert len(listed) == manifest["selected_train_images"] == 2
    assert (output / records[0]["image"]).as_posix() in listed
    assert (output / records[1]["image"]).as_posix() not in listed
    assert manifest["status"] == "EXPERIMENTAL_NONCOMMERCIAL_RESEARCH"
    assert manifest["records"][-1]["rights"]["license_url"].endswith("/by-nc-sa/4.0/")
    assert manifest["records"][:2] == records
    image.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Publisher Git blob"):
        module.build(tmp_path / "tampered-output")
