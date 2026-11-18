import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fetch_module():
    path = Path(__file__).parents[1] / 'scripts/fetch_model.py'
    spec = importlib.util.spec_from_file_location('fetch_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_model_auth_does_not_follow_cross_origin_redirect(fetch_module, monkeypatch, tmp_path):
    data = b'verified-checkpoint'
    artifact = {'filename': 'waste.pt', 'name': 'waste', 'sha256': hashlib.sha256(data).hexdigest(),
                'url': 'https://api.github.com/repos/waste-not-app/waste-not-ai/releases/assets/1'}
    calls, redirects = [], []
    monkeypatch.setenv('GITHUB_TOKEN', 'test-token')
    def build_opener(handler):
        redirects.append(handler)
        return SimpleNamespace(open=lambda request, **_: calls.append(request) or BytesIO(data))
    monkeypatch.setattr(fetch_module.urllib.request, 'build_opener', build_opener)
    fetch_module.fetch(artifact, tmp_path)
    assert (tmp_path / 'waste.pt').read_bytes() == data
    assert calls[0].get_header('Authorization') == 'Bearer test-token'
    redirected = redirects[0].redirect_request(calls[0], None, 302, 'Found', {},
                                              'https://release-assets.githubusercontent.com/signed-asset')
    assert redirected.get_header('Authorization') is None
    downgraded = redirects[0].redirect_request(calls[0], None, 302, 'Found', {},
                                              'http://api.github.com/signed-asset')
    assert downgraded.get_header('Authorization') is None
    public = {**artifact, 'filename': 'general.pt', 'url': 'https://github.com/ultralytics/assets/model.pt'}
    fetch_module.fetch(public, tmp_path)
    assert calls[1].get_header('Authorization') is None


def test_invalid_download_is_removed_and_existing_mismatch_is_preserved(fetch_module, monkeypatch, tmp_path):
    artifact = {'filename': 'waste.pt', 'name': 'waste', 'sha256': hashlib.sha256(b'expected').hexdigest(),
                'url': 'https://example.invalid/model.pt'}
    calls = []
    monkeypatch.setattr(fetch_module.urllib.request, 'build_opener',
                        lambda _: SimpleNamespace(open=lambda *args, **kwargs: calls.append(1) or BytesIO(b'corrupt')))
    with pytest.raises(ValueError, match='checksum'):
        fetch_module.fetch(artifact, tmp_path)
    assert not (tmp_path / 'waste.pt').exists() and not (tmp_path / 'waste.part').exists()
    (tmp_path / 'waste.pt').write_bytes(b'existing-mismatch')
    with pytest.raises(ValueError, match='Existing model'):
        fetch_module.fetch(artifact, tmp_path)
    assert (tmp_path / 'waste.pt').read_bytes() == b'existing-mismatch'
    assert calls == [1]
