"""Prevent a new-data experiment from silently training on the old dataset."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def training(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/train_broad_waste.py"
    spec = importlib.util.spec_from_file_location("training_selection", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    return module


@pytest.mark.parametrize("wrong_yaml,wrong_comparison", [(False, False), (True, False), (False, True)])
def test_explicit_dataset_and_comparison_are_checked_before_expensive_training(
    training, monkeypatch, tmp_path, wrong_yaml, wrong_comparison
):
    import yaml

    data = tmp_path / "new-data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"labels": ["bottle", "can"]}))
    (data / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path / "old-data" if wrong_yaml else data),
                "train": "train.txt",
                "val": "val.txt",
                "test": "test.txt",
            }
        )
    )
    calls = []

    class StopBeforeGPUTraining(Exception):
        pass

    class FakeYOLO:
        def __init__(self, path):
            self.names = {0: "wrong"} if wrong_comparison else {0: "bottle", 1: "can"}

        def add_callback(self, *_):
            pass

        def train(self, **kwargs):
            calls.append(kwargs)
            raise StopBeforeGPUTraining()

    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO, settings=SimpleNamespace(update=lambda _: None))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_broad_waste.py",
            "--data",
            str(data),
            "--device",
            "cpu",
            "--compare-model",
            str(tmp_path / "previous.pt"),
        ],
    )
    if wrong_yaml or wrong_comparison:
        with pytest.raises(ValueError, match="Dataset YAML" if wrong_yaml else "taxonomy mismatch"):
            training.main()
        assert calls == []
    else:
        with pytest.raises(StopBeforeGPUTraining):
            training.main()
        assert calls[0]["data"] == str(data / "data.yaml")


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
