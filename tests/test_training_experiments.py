from dataclasses import asdict, replace
from pathlib import Path
import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml


@pytest.fixture
def training(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    import training_experiments
    return training_experiments


@pytest.fixture
def prepared(training, tmp_path):
    from wastenot_ai.taxonomy import BROAD_LABELS
    data = tmp_path/"data"
    rows = []
    for split in ("train", "val"):
        image = data/f"images/{split}/photo.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (32,32), "green" if split == "train" else "blue").save(image)
        label = data/f"labels/{split}/photo.txt"
        label.parent.mkdir(parents=True)
        label.write_text("5 .5 .5 .5 .5\n")
        rows.append({"image": image.relative_to(data).as_posix(), "split": split,
                     "source": "NewCapture", "prepared_sha256": training.sha(image), "boxes": [[5,.5,.5,.5,.5]],
                     "group": split, "phash": 0 if split == "train" else (1<<63)-1})
        (data/f"{split}.txt").write_text(image.as_posix()+"\n")
    rows.append({"image": "images/test/DO_NOT_OPEN.png", "split": "test", "boxes": [],
                 "group":"test", "phash":(1<<31)-1, "prepared_sha256":"unopened-holdout"})
    training.write(data/"manifest.json", {"labels": list(BROAD_LABELS), "records": rows,
                                        "selected_train_images": 1, "intake_sha256": "reviewed-intake"})
    (data/"data.yaml").write_text(yaml.safe_dump({"path": data.as_posix(), "names": dict(enumerate(BROAD_LABELS)),
                                                 "train": "train.txt", "val": "val.txt", "test": "test.txt"}))
    (data/"test.txt").write_bytes(b"\xff\x00")  # Not readable as UTF-8; no holdout images exist.
    (data/"DATASET_NOTICES.md").write_text("Reviewed attribution")
    training.write(data/"readiness.json", {"status": "READY_FOR_TRAINING_EXPERIMENT",
        "dataset_manifest_sha256": training.sha(data/"manifest.json"), "intake_sha256": "reviewed-intake"})
    model = tmp_path/"initial.pt"
    model.write_bytes(b"pinned baseline fixture")
    options = SimpleNamespace(**asdict(training.TrainingConfig()), data=data, initial_model=model,
        expected_initial_sha256=training.sha(model), name="experiment", device="cpu", evaluate_only=model,
        plan_only=False, stage="B0", b3_selection=None)
    return data, options


@pytest.mark.parametrize("kwargs", [{"imgsz":641},{"batch":0},{"workers":-1},{"seed":-1},
    {"lr0":float("nan")},{"mosaic":1.1},{"freeze":-1},{"close_mosaic":40},{"degrees":181},{"patience":0}])
def test_invalid_parameters_fail_before_model_import(training, kwargs):
    with pytest.raises(ValueError):
        replace(training.TrainingConfig(), **kwargs).validate()


def test_snapshot_ignores_holdout_and_detects_label_changes(training, prepared):
    from wastenot_ai.taxonomy import BROAD_LABELS
    data, _ = prepared
    assert "test" not in training.dataset_snapshot(data, BROAD_LABELS, balanced=True)
    (data/"labels/val/photo.txt").write_text("5 .5 .5 .8 .8\n")
    with pytest.raises(ValueError, match="Labels differ"):
        training.dataset_snapshot(data, BROAD_LABELS, balanced=True)


@pytest.mark.parametrize("fault", ["stale-proof", "wrong-model", "unsafe-name", "B3-without-selection"])
def test_prerequisites_fail_before_model_import(training, prepared, fault):
    data, args = prepared
    if fault == "stale-proof":
        training.write(data/"readiness.json", {"status":"READY_FOR_TRAINING_EXPERIMENT"})
    elif fault == "wrong-model":
        args.expected_initial_sha256 = "wrong"
    elif fault == "unsafe-name":
        args.name = "../other"
    else:
        args.stage, args.evaluate_only, args.lr0 = "B3", None, .00003
    with pytest.raises(ValueError):
        training.experiment(args, "broad", data.parent, model_factory=lambda _: pytest.fail("Model imported too early"))


def test_plan_records_missing_review_without_training(training, prepared):
    data, args = prepared
    args.plan_only = True
    (data/"readiness.json").unlink()
    result = training.experiment(args, "broad", data.parent)
    assert result["status"] == "DEFERRED_DATA_OR_BASELINE" and not result["holdout_opened"]
    assert not (data.parent/"artifacts/training/experiment").exists()


@pytest.fixture
def runtime(training, monkeypatch):
    from wastenot_ai.taxonomy import BROAD_LABELS
    monkeypatch.setattr(training.importlib.metadata, "version", lambda name:"8.4.183")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False),
        set_num_threads=lambda n:None, get_num_threads=lambda:8, version=SimpleNamespace(cuda=None), __version__="fixture"))
    calls = []

    class Model:
        names = dict(enumerate(BROAD_LABELS))
        model = SimpleNamespace(model=list(range(23)))

        def __init__(self, path):
            self.callbacks = {}

        def predict(self, path, **kwargs):
            assert Path(path).parent.name == "val"
            calls.append(("predict",kwargs))
            return [SimpleNamespace(boxes=[])]

        def val(self, **kwargs):
            assert kwargs["split"] == "val"
            assert "test" not in yaml.safe_load(Path(kwargs["data"]).read_text())
            calls.append(("val",kwargs))
            return SimpleNamespace(results_dict={"metrics/mAP50-95(B)":.4},
                box=SimpleNamespace(ap_class_index=[5],class_result=lambda i:(.5,.5,.5,.4)))

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            assert "test" not in yaml.safe_load(Path(kwargs["data"]).read_text())
            calls.append(("train",kwargs))
            run = Path(kwargs["project"])/kwargs["name"]
            best = run/"weights/best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"unstripped best")
            trainer = SimpleNamespace(epoch=2,fitness=.7,best_fitness=.7,best=best,
                batch_size=kwargs["batch"],accumulate=4,metrics={"map":.7})
            self.callbacks["on_model_save"](trainer)
            self.callbacks["on_fit_epoch_end"](trainer)
            (run/"results.csv").write_text("epoch,metrics/mAP50-95(B)\n1,0.9\n2,0.4\n3,0.7\n")
            (run/"args.yaml").write_text(yaml.safe_dump(kwargs))
            best.write_bytes(b"stripped best")
            self.trainer = SimpleNamespace(save_dir=run)
    return Model, calls


def test_evaluation_only_never_trains_or_opens_holdout(training, prepared, runtime):
    data, args = prepared
    result = training.experiment(args, "broad", data.parent, runtime[0])
    assert result["status"] == "COMPLETED_VALIDATION_ONLY" and result["training"]["best_epoch"] is None
    assert not any(k == "train" for k,v in runtime[1])
    assert sum(k == "val" for k,v in runtime[1]) == 1
    assert result["comparison"]["baseline"] == result["comparison"]["candidate"]


def test_parameters_and_best_epoch_follow_the_actual_trainer(training, prepared, runtime):
    data, args = prepared
    args.stage, args.evaluate_only, args.imgsz, args.batch = "extra", None, 768, 8
    result = training.experiment(args, "broad", data.parent, runtime[0])
    kwargs = next(v for k,v in runtime[1] if k == "train")
    assert kwargs["imgsz"] == 768 and kwargs["batch"] == 8 and kwargs["freeze"] == 10
    assert kwargs["mosaic"] == .5 and kwargs["seed"] == 42 and kwargs["optimizer"] == "AdamW"
    assert all(v["imgsz"] == 768 for k,v in runtime[1] if k in {"val","predict"})
    assert result["training"]["best_epoch"] == 3  # CSV peak is epoch 1, deliberately different.
    assert result["training"]["final_checkpoint_sha256"] != result["training"]["checkpoint_selection"]["pre_strip_sha256"]
    with pytest.raises(ValueError,match="Run exists"):
        training.experiment(args,"broad",data.parent,runtime[0])


def test_freeze_cannot_disable_every_layer(training, prepared, runtime):
    data, args = prepared
    args.stage, args.freeze = "extra", 23
    with pytest.raises(ValueError,match="Freeze"):
        training.experiment(args,"broad",data.parent,runtime[0])


def test_general_runner_preserves_80_labels_without_opening_holdout(training, prepared, runtime):
    from wastenot_ai.taxonomy import COCO_LABELS
    data, args = prepared
    manifest = json.loads((data/"manifest.json").read_text())
    manifest["labels"] = list(COCO_LABELS)
    for row in manifest["records"]:
        if row["split"] in {"train","val"}:
            row["boxes"][0][0] = 67
            (data/f"labels/{row['split']}/photo.txt").write_text("67 .5 .5 .5 .5\n")
    training.write(data/"manifest.json",manifest)
    config = yaml.safe_load((data/"data.yaml").read_text())
    config["names"] = dict(enumerate(COCO_LABELS))
    (data/"data.yaml").write_text(yaml.safe_dump(config))
    training.write(data/"readiness.json",{"status":"READY_FOR_TRAINING_EXPERIMENT",
        "dataset_manifest_sha256":training.sha(data/"manifest.json"),"intake_sha256":"reviewed-intake"})
    runtime[0].names = dict(enumerate(COCO_LABELS))
    result = training.experiment(args,"general",data.parent,runtime[0])
    assert len(result["labels"]) == 80 and result["labels"][67] == "cell phone"
    assert result["comparison"]["candidate"]["fixed_validation"]["per_class"]["cell phone"]["support"] == 1


def test_fixed_latency_percentile_does_not_fall_below_median(training, prepared, runtime, monkeypatch):
    import train_broad_waste
    data, _ = prepared
    paths = []
    for i in range(3):
        image = data/f"images/val/p{i}.png"
        image.write_bytes((data/"images/val/photo.png").read_bytes())
        (data/f"labels/val/p{i}.txt").write_text("5 .5 .5 .5 .5\n")
        paths.append(image.as_posix())
    (data/"val.txt").write_text("\n".join(paths)+"\n")
    times = iter([0,.001,.01,.014,.02,.026])
    monkeypatch.setattr(train_broad_waste,"perf_counter",lambda:next(times))
    result = train_broad_waste.evaluate_fixed(runtime[0]("fixture"),data,list(runtime[0].names.values()),"cpu",dict(enumerate(range(10))))
    assert result["latency_p50_ms"] == pytest.approx(5)
    assert result["latency_p95_ms"] == pytest.approx(5.9)
