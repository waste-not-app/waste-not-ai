import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_resolution_benchmark import summary


@pytest.fixture
def workflow(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    import benchmark_resolutions
    return benchmark_resolutions


def test_frozen_receipts_bind_data_and_candidate_before_model_import(workflow,tmp_path,monkeypatch):
    registry, specs = {}, {}
    for branch in ("broad","general"):
        weights = tmp_path/(branch+".pt")
        weights.write_bytes(branch.encode())
        artifact = {"sha256":workflow.sha(weights)}
        if branch == "broad":
            registry.update(artifact)
        else:
            registry["general_model"] = artifact
        receipt = tmp_path/(branch+".json")
        workflow.write(receipt,{"branch":branch,"status":"VALIDATION_STABLE_PENDING_RESOLUTION_REVIEW",
            "holdout_opened":False,"selected_candidate_sha256":workflow.sha(weights),
            "input_sha256":{"manifest":"frozen","initial_model":artifact["sha256"]}})
        specs[branch] = (tmp_path/branch,weights,receipt)
    workflow.write(tmp_path/"model.json",registry)
    monkeypatch.setattr(workflow,"dataset_snapshot",lambda *args,**kwargs:{"manifest":"frozen"})
    assert set(workflow.freeze_components(specs,tmp_path)) == {"broad","general"}
    specs["broad"][1].write_bytes(b"changed")
    with pytest.raises(ValueError,match="differ"):
        workflow.freeze_components(specs,tmp_path)


def specs_with_shared_image(workflow,tmp_path):
    from wastenot_ai.taxonomy import BROAD_LABELS,COCO_LABELS
    specs = {}
    for branch,labels,cls in (("broad",BROAD_LABELS,0),("general",COCO_LABELS,39)):
        data = tmp_path/branch
        workflow.write(data/"manifest.json",{"labels":list(labels),"records":[{
            "image":"images/val/shared.png","split":"val","prepared_sha256":"same-image",
            "phash":0,"group":branch+"-object","boxes":[[cls,.5,.5,.4,.4]]}]})
        specs[branch] = (data,None,None)
    return specs


def test_shared_validation_photo_is_not_double_counted(workflow,tmp_path):
    specs = specs_with_shared_image(workflow,tmp_path)
    manifest = workflow.validation_manifest(specs,tmp_path)
    assert manifest["usage"] == "VALIDATION_ONLY" and len(manifest["records"]) == 1
    assert len(manifest["records"][0]["truth"]) == 1
    assert len(manifest["records"][0]["evaluated_classes"]) == 88


def test_conflicting_annotations_and_cross_model_sessions_are_rejected(workflow,tmp_path):
    specs = specs_with_shared_image(workflow,tmp_path)
    path = specs["general"][0]/"manifest.json"
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed["records"][0]["boxes"][0][1] = .6
    workflow.write(path,changed)
    with pytest.raises(ValueError,match="conflicting"):
        workflow.validation_manifest(specs,tmp_path)
    broad_path = specs["broad"][0]/"manifest.json"
    broad = json.loads(broad_path.read_text())
    broad["records"][0]["session_id"] = "same-session"
    workflow.write(broad_path,broad)
    row = original["records"][0]
    row.update(split="train",session_id="same-session",prepared_sha256="other",phash=(1<<63)-1)
    workflow.write(path,original)
    with pytest.raises(ValueError,match="leakage"):
        workflow.validation_manifest(specs,tmp_path)


@pytest.mark.parametrize("budget_ok",[False,True])
def test_960_only_runs_after_768_gate_and_remeasures_640(workflow,tmp_path,monkeypatch,budget_ok):
    specs = {branch:(tmp_path, tmp_path/(branch+".pt"), tmp_path/(branch+"-receipt.json")) for branch in ("broad","general")}
    monkeypatch.setattr(workflow,"freeze_components",lambda *args:{"frozen":True})
    manifest = {"usage":"VALIDATION_ONLY","records":[],"target_classes":["bottle"]}
    monkeypatch.setattr(workflow,"validation_manifest",lambda *args:manifest)
    monkeypatch.setattr(workflow,"load_manifest",lambda *args,**kwargs:manifest)
    monkeypatch.setattr(workflow,"prepare_inputs",lambda *args:[])
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(set_num_threads=lambda n:None,get_num_threads=lambda:8,__version__="fixture"))
    monkeypatch.setattr(workflow.importlib.metadata,"version",lambda name:"8.4.183")
    monkeypatch.setattr(workflow,"load_detector",lambda *args,**kwargs:SimpleNamespace(artifact={"size":kwargs["input_size"]}))
    calls = []
    def stage(detectors,*args,**kwargs):
        calls.append(sorted(detectors))
        values = {"640":summary(),"768":summary(.6,120 if budget_ok else 126)}
        if 960 in detectors:
            values["960"] = summary(.65,124)
        return {"results":{"browser_jpeg90":values},"models":{str(k):v.artifact for k,v in detectors.items()}}
    monkeypatch.setattr(workflow,"benchmark_stage",stage)
    monkeypatch.setattr(workflow,"component_ap",lambda *args:{"separate_components":True})
    result = workflow.run(specs,tmp_path/"output",root=tmp_path)
    assert calls == ([[640,768],[640,768,960]] if budget_ok else [[640,768]])
    assert result["selection"]["selected_input_size"] == (960 if budget_ok else 640)
    assert result["http"]["status"] == "NOT_MEASURED"
    assert result["active_input_size_unchanged"] == 640


def test_http_probe_requires_matching_remote_profile_and_keeps_token_out_of_report(workflow,monkeypatch):
    artifact = {"sha256":"expected","model_version":"profile768"}
    seen = []
    class Client:
        def __init__(self,**kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self,*args):
            pass

        def post(self,url,content,headers):
            seen.append(headers)
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{"data":{
                "model":artifact,"image":{"width":100,"height":50}}})
    monkeypatch.setattr(workflow.httpx,"Client",Client)
    inputs = [{"content":b"image","transmitted_size":(100,50)}]
    report = workflow.measure_http({"768":"http://127.0.0.1/internal/v1/detect"},{"768":artifact},inputs,token="dummy-token")
    assert len(seen) == 8 and report["status"] == "MEASURED_SEPARATELY"
    assert "dummy-token" not in json.dumps(report)
    with pytest.raises(ValueError,match="expected resolution"):
        workflow.measure_http({"768":"http://127.0.0.1/internal/v1/detect"},
                              {"768":{"sha256":"other","model_version":"other"}},inputs,token="dummy-token")


def test_remote_plaintext_endpoint_is_rejected_before_upload(workflow):
    with pytest.raises(ValueError,match="HTTPS"):
        workflow.measure_http({"640":"http://example.org/internal/v1/detect"},{"640":{}},[],token="dummy")
