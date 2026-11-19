import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from wastenot_ai import providers, recommendation, recommendation_worker


@pytest.mark.parametrize('seconds', [15, 17, 17.01, 120])
def test_recommendation_deadline_fits_go_header_timeout(monkeypatch, seconds):
    spec = importlib.util.spec_from_file_location('runtime_entry', Path(__file__).parents[1] / 'serve.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(providers, 'GeminiProvider', lambda: SimpleNamespace(timeout=seconds))
    monkeypatch.setattr(recommendation, 'load_guides', lambda _: [])
    monkeypatch.delenv('AI_GUIDES_PATH', raising=False)
    calls = []
    monkeypatch.setattr(recommendation_worker, 'create_recommendation_app', lambda **values: calls.append(values))
    if seconds > 17:
        with pytest.raises(ValueError, match='Android/Go API deadlines'):
            runtime.application('recommendation')
        assert calls == []
    else:
        runtime.application('recommendation')
        assert seconds < calls[0]['timeout'] < 20 < 23
        assert calls[0]['demo_guides'] is False


@pytest.mark.parametrize('combined', [False, True])
@pytest.mark.parametrize('kind', ['detection', 'demo'])
def test_detection_manifest_checks_both_models_and_warms_up_before_serving(monkeypatch, tmp_path, combined, kind):
    from wastenot_ai import demo, detector, worker

    spec = importlib.util.spec_from_file_location('runtime_bundle', Path(__file__).parents[1] / 'serve.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.__file__ = str(tmp_path / 'serve.py')
    manifest = {
        'filename': 'waste.pt',
        'sha256': hashlib.sha256(b'waste').hexdigest(),
        'input_shape': 'square',
        'model_version': 'test-combined-model' if combined else 'test-single-model',
    }
    (tmp_path / 'waste.pt').write_bytes(b'waste')
    if combined:
        manifest['general_model'] = {'filename': 'general.pt', 'sha256': hashlib.sha256(b'general').hexdigest()}
        (tmp_path / 'general.pt').write_bytes(b'general')
    (tmp_path / 'model.json').write_text(json.dumps(manifest))
    monkeypatch.setenv('AI_MODEL_DIRECTORY', str(tmp_path))
    monkeypatch.setenv('GEMINI_API_KEY', 'fixture-provider-key')
    monkeypatch.setitem(
        sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
    )
    calls = []
    loaded = SimpleNamespace(
        artifact={'model_version': manifest['model_version']}, detect=lambda raw: calls.append(('warmup', raw))
    )

    def load(path, **options):
        calls.append(('load', path, options))
        return loaded

    monkeypatch.setattr(detector, 'load_detector', load)
    monkeypatch.setattr(worker, 'create_app', lambda **options: calls.append(('serve', options)))
    monkeypatch.setattr(demo, 'create_demo_app', lambda **options: calls.append(('serve', options)))
    runtime.application(kind)
    assert calls[0] == (
        'load',
        tmp_path / 'waste.pt',
        {'device': 'cpu', 'input_shape': 'square', 'general_model_path': tmp_path / 'general.pt' if combined else None},
    )
    assert calls[1][0] == 'warmup' and calls[1][1].startswith(b'\x89PNG')
    expected = {'detector': loaded, 'demo_guides': True} if kind == 'demo' else {'detector': loaded}
    assert calls[2] == ('serve', expected)
    assert ('GEMINI_API_KEY' in runtime.os.environ) == (kind == 'demo')
    calls.clear()
    (tmp_path / ('general.pt' if combined else 'waste.pt')).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        runtime.application(kind)
    assert calls == []


def test_legacy_single_model_environment_remains_supported(monkeypatch, tmp_path):
    from wastenot_ai import detector, worker

    spec = importlib.util.spec_from_file_location('runtime_legacy', Path(__file__).parents[1] / 'serve.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    monkeypatch.delenv('AI_MODEL_DIRECTORY', raising=False)
    path = tmp_path / 'baseline.pt'
    path.write_bytes(b'baseline')
    monkeypatch.setenv('AI_MODEL_PATH', str(path))
    monkeypatch.setenv('AI_MODEL_SHA256', hashlib.sha256(b'baseline').hexdigest())
    monkeypatch.setitem(
        sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
    )
    calls = []
    loaded = SimpleNamespace(detect=lambda _: None)
    monkeypatch.setattr(detector, 'load_detector', lambda *args, **options: calls.append((args, options)) or loaded)
    monkeypatch.setattr(worker, 'create_app', lambda **_: None)
    runtime.application('detection')
    assert calls == [((path,), {'device': 'cpu', 'input_shape': 'adaptive', 'general_model_path': None})]
