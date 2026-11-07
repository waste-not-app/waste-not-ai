import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def focus(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("focus_data", scripts / "prepare_focus_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cardboard_builder_preserves_complete_manifest_when_parent_returns_summary(focus, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    (cache / "cardboard").mkdir(parents=True)
    (cache / "cardboard/acquisition.json").write_text(json.dumps({"records": []}))
    (cache / "cardboard/review.json").write_text("[]")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"records": []}))
    output = tmp_path / "dataset"
    monkeypatch.setattr(focus, "CACHE", cache)
    monkeypatch.setattr(focus, "BENCHMARK", benchmark)
    monkeypatch.setattr(focus, "WASTE", output)
    monkeypatch.setattr(focus, "ROOT", tmp_path)
    monkeypatch.setattr(focus, "frozen_images", lambda: ({"records": []}, {"records": []}))

    def parent_builder(*args, **kwargs):
        output.mkdir()
        manifest = {"records": [{"source_id": "reserved"}], "new_images": 1, "active_instances_per_split": {}}
        (output / "manifest.json").write_text(json.dumps(manifest))
        (output / "DATASET_NOTICES.md").write_text("Parent notices")
        return {key: value for key, value in manifest.items() if key != "records"}

    monkeypatch.setattr(focus.expansion, "build_dataset", parent_builder)
    focus.build_cardboard()
    result = json.loads((output / "manifest.json").read_text())
    assert result["records"] == [{"source_id": "reserved"}]
    assert result["benchmark_sha256"]


def test_focus_selection_retains_other_coco_objects_and_never_duplicates_an_image(focus):
    phone = focus.COCO_LABELS.index("cell phone")
    remote = focus.COCO_LABELS.index("remote")
    person = focus.COCO_LABELS.index("person")
    boxes = [[phone,.5,.5,.4,.4], [remote,.1,.1,.1,.1], [person,.5,.5,.9,.9]]
    rows = [{"source_id":"both", "boxes":boxes},
            {"source_id":"remote", "boxes":[[remote,.5,.5,.4,.4]]}]
    selected = focus.select_candidates(rows, [("cell phone",1), ("remote",1)])
    assert [row["source_id"] for row in selected] == ["both", "remote"]
    assert selected[0]["boxes"] == boxes


def test_shared_cardboard_negative_cannot_leak_waste_holdout_to_general_training(focus):
    rows = [{"source_id":"leak", "source_split":"validation", "intended_split":"train", "phash":0},
            {"source_id":"safe", "source_split":"validation", "intended_split":"train", "phash":2**40-1}]
    waste = [{"split":"test", "phash":0, "image":"reserved.jpg"},
             {"split":"train", "phash":2**40-1, "image":"train.jpg"}]
    assigned, rejected = focus.assign_general_splits(rows, waste)
    assert [row["source_id"] for row in assigned] == ["safe"]
    assert rejected[0]["source_id"] == "leak" and rejected[0]["waste_split"] == "test"


def test_general_holdout_cannot_reuse_another_models_training_photo(focus):
    rows = [{"source_id":"validation", "source_split":"val2017", "selection_stratum":"phone", "phash":0}]
    assigned, rejected = focus.assign_general_splits(rows, [{"split":"train", "phash":0, "image":"train.jpg"}])
    assert assigned == [] and rejected[0]["source_id"] == "validation"


def test_dataset_verifier_requires_explicit_review_for_empty_negative_labels(monkeypatch, tmp_path):
    import hashlib
    from PIL import Image

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("focus_verifier", scripts / "verify_broad_data.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    image = tmp_path / "images/train/negative.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32,32), "white").save(image)
    label = tmp_path / "labels/train/negative.txt"
    label.parent.mkdir(parents=True)
    label.write_text("")
    row = {"image":"images/train/negative.jpg", "split":"train", "source":"OpenImages",
           "group":"negative", "phash":0, "boxes":[], "kind":"VERIFIED_NEGATIVE",
           "prepared_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
           "rights":{"accepted":True,"license_url":"fixture","source_page":"fixture"},
           "visual_review":{"decision":"APPROVED","general_negative":True}}
    manifest = {"labels":["bottle"],"records":[row]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    for split in ("train","val","test"):
        (tmp_path / (split+".txt")).write_text(image.as_posix()+"\n" if split=="train" else "")
    monkeypatch.setattr("sys.argv", ["verify", "--data",str(tmp_path), "--output",str(tmp_path/"result.json")])
    verifier.main()
    row["visual_review"]["general_negative"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        verifier.main()
