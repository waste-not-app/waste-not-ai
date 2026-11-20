"""The legacy general dataset validator retains taxonomy and membership checks.

Runner preflight and holdout isolation are tested in test_training_experiments.py.
"""

import importlib.util
import json
from pathlib import Path

import pytest


def test_general_focus_dataset_requires_native_taxonomy_and_complete_split_lists(monkeypatch, tmp_path):
    import yaml

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("general_training", scripts / "train_general_focus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = list(module.COCO_LABELS)
    records = [{"split": split, "image": f"images/{split}/photo.jpg"} for split in ("train", "val", "test")]
    manifest = {"labels": labels, "records": records}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "path": str(tmp_path),
        "names": dict(enumerate(labels)),
        **{split: split + ".txt" for split in ("train", "val", "test")},
    }
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(config))
    for row in records:
        (tmp_path / (row["split"] + ".txt")).write_text((tmp_path / row["image"]).as_posix() + "\n")
    assert module.validate_dataset(tmp_path, labels) == manifest
    manifest["labels"] = list(reversed(labels))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="native COCO taxonomy"):
        module.validate_dataset(tmp_path, labels)
    manifest["labels"] = labels
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "train.txt").write_text((tmp_path / records[-1]["image"]).as_posix() + "\n")
    with pytest.raises(ValueError, match="train list"):
        module.validate_dataset(tmp_path, labels)
