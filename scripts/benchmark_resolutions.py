"""Compare frozen issue #10 components on reviewed validation data, without activating a profile."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from urllib.parse import urlparse

import httpx
import yaml

from training_experiments import dataset_snapshot, sha, write
from prepare_external_focus import phash_neighbors
from wastenot_ai.camera_benchmark import browser_input, load_manifest
from wastenot_ai.detector import load_detector
from wastenot_ai.resolution_benchmark import (benchmark_stage, cache_identity, distribution, prepare_inputs,
                                           select_resolution, PROFILES)
from wastenot_ai.taxonomy import BROAD_LABELS, COCO_LABELS, COMBINED_LABELS

ROOT = Path(__file__).resolve().parents[1]


def freeze_components(specs, root=ROOT):
    registry = json.loads((root/"model.json").read_text("utf-8"))
    frozen = {}
    for branch, (data,weights,receipt_path) in specs.items():
        artifact = registry if branch == "broad" else registry["general_model"]
        receipt = json.loads(receipt_path.read_text("utf-8"))
        if receipt.get("branch") != branch or receipt.get("holdout_opened") is not False:
            raise ValueError("Invalid frozen component receipt")
        status = receipt.get("status")
        if status == "VALIDATION_STABLE_PENDING_RESOLUTION_REVIEW":
            expected = receipt.get("selected_candidate_sha256")
        elif status in {"KEEP_BASELINE","KEEP_BASELINE_UNSTABLE"}:
            expected = artifact["sha256"]
        else:
            raise ValueError("Component selection/stability is not complete")
        labels = BROAD_LABELS if branch == "broad" else COCO_LABELS
        snapshot = dataset_snapshot(data.resolve(),labels,balanced=branch == "broad")
        if {**snapshot,"initial_model":artifact["sha256"]} != receipt.get("input_sha256") or sha(weights) != expected:
            raise ValueError("Candidate weights or validation dataset differ from frozen receipt")
        frozen[branch] = {"weights_sha256":expected,"receipt_sha256":sha(receipt_path),"dataset_snapshot":snapshot}
    if set(frozen) != {"broad","general"}:
        raise ValueError("Both frozen model components are required")
    return frozen


def validation_manifest(specs, root=ROOT):
    grouped, protection = {}, []
    for branch,(data,_,_) in specs.items():
        manifest = json.loads((data/"manifest.json").read_text("utf-8"))
        protection.extend(manifest["records"])
        for row in manifest["records"]:
            if row["split"] != "val":
                continue
            checksum = row["prepared_sha256"]
            truth = [{"label":manifest["labels"][int(c)],"bbox":[x-w/2,y-h/2,x+w/2,y+h/2]} for c,x,y,w,h in row["boxes"]]
            # Legacy source edge rounding is allowed by its verifier; normalize GT edges only for scoring.
            for box in truth:
                box["bbox"] = [max(0,min(1,v)) for v in box["bbox"]]
            scope = list(manifest["labels"])
            if checksum in grouped:
                existing = grouped[checksum]
                shared = set(scope) & set(existing["evaluated_classes"])
                canonical = lambda rows: sorted((r["label"],tuple(round(v,6) for v in r["bbox"])) for r in rows if r["label"] in shared)
                if canonical(existing["truth"]) != canonical(truth):
                    raise ValueError("Shared validation image has conflicting annotations")
                existing["truth"].extend(r for r in truth if r["label"] not in shared)
                existing["evaluated_classes"] = sorted(set(scope)|set(existing["evaluated_classes"]))
                existing["source_groups"].append(row["group"])
                if row.get("session_id"):
                    existing["source_groups"].append("session:" + row["session_id"])
                continue
            grouped[checksum] = {"id":checksum,"sha256":checksum,
                "image":(data/row["image"]).resolve().relative_to(root.resolve()).as_posix(),
                "group":row["group"],"source_groups":[row["group"]],"source":branch,
                "split":"val","phash":row["phash"],"evaluated_classes":scope,"truth":truth}
            if row.get("session_id"):
                grouped[checksum]["source_groups"].append("session:" + row["session_id"])
    groups, hashes, sessions = {}, {}, {}
    for row in protection:
        for field, table in (("group",groups),("prepared_sha256",hashes),("session_id",sessions)):
            if field == "session_id" and not row.get(field):
                continue
            if table.setdefault(row[field],row["split"]) != row["split"]:
                raise ValueError("Cross-model dataset leakage")
    for i,j in phash_neighbors(protection):
        if protection[i]["split"] != protection[j]["split"]:
            raise ValueError("Cross-model near-duplicate split leakage")
    return {"usage":"VALIDATION_ONLY","target_classes":list(COMBINED_LABELS),"records":list(grouped.values()),
            "limitations":["Operational scores honor each source's annotation scope.",
                            "Synthetic dim input is another view of the same images, not independent support."]}


def component_ap(detectors, specs, output):
    configs = {}
    for branch, (data, _, _) in specs.items():
        work = output / f"ap-input-{branch}"
        images, labels = work / "images", work / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        listed = []
        for index, line in enumerate((data / "val.txt").read_text("utf-8").splitlines()):
            source = Path(line)
            target = images / f"{index}.jpg"
            target.write_bytes(browser_input(source))
            (labels / f"{index}.txt").write_bytes((data / "labels" / "val" / source.with_suffix(".txt").name).read_bytes())
            listed.append(target.as_posix())
        validation_list = work / "val.txt"
        validation_list.write_text("\n".join(listed) + "\n", encoding="utf-8")
        configs[branch] = output / f"ap-{branch}-validation.yaml"
        configs[branch].write_text(yaml.safe_dump({"path": data.resolve().as_posix(),
            "names": dict(enumerate(BROAD_LABELS if branch == "broad" else COCO_LABELS)),
            "train": (data/"train.txt").resolve().as_posix(), "val": validation_list.resolve().as_posix()}), encoding="utf-8")
    reports = {}
    for size,combined in detectors.items():
        reports[str(size)] = {}
        for branch,detector in (("broad",combined.waste),("general",combined.general)):
            value = detector.model.val(data=str(configs[branch]),split="val",device="cpu",batch=1,workers=0,
                imgsz=size,rect=False,conf=.001,iou=.7,max_det=50,plots=False,
                project=str(output/"component-ap"),name=f"{branch}-{size}")
            per_class = []
            for index,cls in enumerate(value.box.ap_class_index):
                p,r,ap50,ap = value.box.class_result(index)
                per_class.append({"label":detector.labels[int(cls)],"precision":float(p),"recall":float(r),
                                  "map50":float(ap50),"map50_95":float(ap)})
            reports[str(size)][branch] = {"split":"val","metrics":{k:float(v) for k,v in value.results_dict.items()},
                                          "per_class":per_class}
    return reports


def measure_http(endpoints, artifacts, inputs, *, repeats=3, warmup=5, token=None):
    if not endpoints:
        return {"status":"NOT_MEASURED","reason":"No dedicated profile endpoints provided"}
    if set(endpoints) != set(artifacts) or repeats < 3 or warmup < 5 or not token:
        raise ValueError("HTTP measurement requires all matching profile endpoints, a token, warmup>=5 and repeats>=3")
    for url in endpoints.values():
        parsed = urlparse(url)
        if parsed.scheme not in {"http","https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Invalid HTTP profile endpoint")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1","localhost","::1"}:
            raise ValueError("Remote profile endpoints require HTTPS")
    results = {}
    with httpx.Client(timeout=30,follow_redirects=False) as client:
        for size,url in endpoints.items():
            times = []
            samples = [inputs[0]]*warmup + [item for item in inputs for _ in range(repeats)]
            for index,item in enumerate(samples):
                started = perf_counter()
                response = client.post(url,content=item["content"],headers={"Authorization":"Bearer "+token,"Content-Type":"image/jpeg"})
                response.raise_for_status()
                elapsed = (perf_counter()-started)*1000
                data = response.json()["data"]
                if (data["model"]["sha256"] != artifacts[size]["sha256"]
                        or data["model"]["model_version"] != artifacts[size]["model_version"]
                        or data["image"] != dict(zip(("width","height"),item["transmitted_size"]))):
                    raise ValueError("HTTP worker does not serve the expected resolution/image context")
                if index >= warmup:
                    times.append(elapsed)
            results[size] = distribution(times)
    return {"status":"MEASURED_SEPARATELY","results":results,
            "scope":"HTTP upload/server/response wall clock; includes remote inference; not a subtraction-based overhead estimate"}


def run(specs, output, *, threads=8, repeats=3, warmup=5, endpoints=None, root=ROOT):
    if output.exists() or threads < 1 or repeats < 3 or warmup < 5:
        raise ValueError("Use a fresh output, positive threads, >=3 repeats and >=5 warmups")
    frozen = freeze_components(specs,root)
    manifest = validation_manifest(specs,root)
    output.mkdir(parents=True)
    manifest_path = output/"validation-manifest.json"
    write(manifest_path,manifest)
    manifest_sha = sha(manifest_path)
    manifest = load_manifest(manifest_path,root,allowed_usage=("VALIDATION_ONLY",))
    inputs = prepare_inputs(manifest,root)
    import torch
    if importlib.metadata.version("ultralytics") != "8.4.183":
        raise ValueError("Use pinned Ultralytics 8.4.183")
    torch.set_num_threads(threads)
    environment = {"platform":platform.platform(),"processor":platform.processor(),"cpu_count":os.cpu_count(),
        "torch":torch.__version__,"python":platform.python_version(),"threads":torch.get_num_threads(),
        "ultralytics":importlib.metadata.version("ultralytics"),"Pillow":importlib.metadata.version("Pillow")}
    def create(size):
        return load_detector(specs["broad"][1],"cpu","square",specs["general"][1],input_size=size)
    detectors = {size:create(size) for size in (640,768)}
    first = benchmark_stage(detectors,inputs,manifest["target_classes"],repeats=repeats,warmup=warmup)
    first["selection"] = select_resolution(first)
    write(output/"stage-640-768.json",first)
    stages = [first]
    if first["selection"]["decisions"]["768"]["accepted_for_further_evaluation"]:
        detectors[960] = create(960)
        # A fresh 640/768 control in the same stage keeps the 960 comparison matched.
        stages.append(benchmark_stage(detectors,inputs,manifest["target_classes"],repeats=repeats,warmup=warmup))
        write(output/"stage-640-768-960.json",stages[-1])
    final = stages[-1]
    selected = select_resolution(final)
    ap = component_ap(detectors,specs,output)
    http = measure_http(endpoints or {},final["models"],inputs,repeats=repeats,warmup=warmup,token=os.getenv("AI_SERVICE_TOKEN"))
    if frozen != freeze_components(specs,root) or sha(manifest_path) != manifest_sha:
        raise ValueError("Frozen experiment inputs changed during measurement")
    report = {"status":"VALIDATION_EXPERIMENT_PENDING_FINAL_REVIEW","created_at":datetime.now(timezone.utc).isoformat(),
        "environment":environment,"frozen_components":frozen,"manifest_sha256":sha(manifest_path),
        "cache_sha256":cache_identity(sha(manifest_path),final["models"],inputs,environment),
        "stages":stages,"component_validation_ap":ap,"http":http,"selection":selected,
        "promoted":False,"holdout_opened":False,"active_input_size_unchanged":640}
    write(output/"report.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","run"))
    for branch,dataset in (("broad","broad-waste-issue8-v5"),("general","general-focus-issue9-v2")):
        parser.add_argument(f"--{branch}-data",type=Path,default=ROOT/"data/local"/dataset)
        parser.add_argument(f"--{branch}-receipt",type=Path,default=ROOT/f"artifacts/training/{branch}-stability.json")
        parser.add_argument(f"--{branch}-weights",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--http-endpoint",action="append",default=[],help="640=https://.../internal/v1/detect; repeat for each measured size")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 3 or args.warmup < 5:
        parser.error("Positive threads, >=3 repeats and >=5 warmups required")
    registry = json.loads((ROOT/"model.json").read_text("utf-8"))
    specs = {branch:(getattr(args,branch+"_data"),getattr(args,branch+"_weights") or ROOT/"artifacts/verified-model-bundle"/artifact["filename"],
                     getattr(args,branch+"_receipt")) for branch,artifact in (("broad",registry),("general",registry["general_model"]))}
    if args.command == "plan":
        try:
            freeze_components(specs)
            status = "READY_NOT_EXECUTED"
        except (OSError,ValueError,KeyError):
            status = "DEFERRED_FROZEN_CANDIDATES_OR_REVIEWED_DATA"
        result = {"status":status,"initial_sizes":[640,768],"conditional_size":960,
            "input_profiles":PROFILES,
            "threads":args.threads,"repeats":args.repeats,"warmup":args.warmup,
            "requires":"issue #10 frozen components and reviewed validation datasets",
            "cpu_p95_budget_ratio":1.25,"active_input_size":640,"promoted":False,"holdout_opened":False}
        write(args.output,result)
    else:
        endpoints = dict(item.split("=",1) for item in args.http_endpoint)
        result = run(specs,args.output.resolve(),threads=args.threads,repeats=args.repeats,warmup=args.warmup,endpoints=endpoints)
    print(json.dumps({"status":result["status"],"output":str(args.output)}))


if __name__ == "__main__":
    main()
