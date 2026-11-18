import gzip
import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


deploy = module('deploy_ai', 'deploy/deploy.py')
packager = module('package_ai', 'scripts/package_release.py')


@pytest.fixture
def payload(tmp_path):
    output = tmp_path / 'release.tar.gz'
    packager.package(output, 'a' * 40)
    return output.read_bytes()


def archive(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for name, raw in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            tar.addfile(entry, io.BytesIO(raw))
    return buffer.getvalue()


def test_packaged_runtime_excludes_keys_models_training_and_tests(payload):
    commit, files = deploy.parse_release(payload)
    assert commit == 'a' * 40
    assert 'src/wastenot_ai/worker.py' in files
    assert all(
        not name.endswith(('.pt', '.env', '.pyc')) and not name.startswith(('tests/', 'data/')) for name in files
    )


@pytest.mark.parametrize('name', ['../escape', 'src/wastenot_ai/../../escape', '/etc/passwd'])
def test_traversal_is_rejected(name):
    with pytest.raises(ValueError, match='Unexpected release member'):
        deploy.parse_release(archive({name: b'x'}))


def test_tampered_source_is_rejected(payload):
    _, files = deploy.parse_release(payload)
    files['serve.py'] += b'\n# changed\n'
    with pytest.raises(ValueError, match='checksum'):
        deploy.parse_release(archive(files))


def test_symlink_member_is_rejected():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        member = tarfile.TarInfo('serve.py')
        member.type = tarfile.SYMTYPE
        member.linkname = '/etc/passwd'
        tar.addfile(member)
    with pytest.raises(ValueError):
        deploy.parse_release(buffer.getvalue())


def test_compressed_expansion_is_bounded_before_tar_parsing():
    with patch.object(deploy, 'MAX_EXPANDED', 1024), pytest.raises(ValueError, match='Expanded release'):
        deploy.parse_release(gzip.compress(b'x' * 1025))


def test_peer_contract_drift_is_rejected(tmp_path):
    checker = module('check_contracts', 'scripts/check_contracts.py')
    bundle = tmp_path / 'contracts'
    bundle.mkdir()
    (bundle / 'api.json').write_text('{}')
    pin = tmp_path / 'pin.json'
    pin.write_text(json.dumps({'sha256': {'api.json': hashlib.sha256(b'{}').hexdigest()}}))
    checker.check(bundle, pin)
    peer = tmp_path / 'peer'
    peer.mkdir()
    (peer / 'api.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Peer contract'):
        checker.check(bundle, pin, peer)


def test_changed_dependencies_fail_before_activation(tmp_path, payload):
    _, files = deploy.parse_release(payload)
    (tmp_path / 'ai-requirements.sha256').write_text('0' * 64)
    with patch.object(deploy, 'CONFIG', tmp_path), patch.object(deploy, 'link') as link:
        with pytest.raises(ValueError, match='locked CPU environment'):
            deploy.deploy(payload)
        link.assert_not_called()


@pytest.mark.parametrize('restart_fails', [True, False])
def test_failed_activation_restores_previous_source_and_service_paths(tmp_path, payload, restart_fails):
    root = tmp_path / 'ai'
    previous = root / 'releases' / ('b' * 40)
    previous.mkdir(parents=True)
    (root / 'current').symlink_to(previous)
    units = tmp_path / 'units'
    with (
        patch.object(deploy, 'ROOT', root),
        patch.object(deploy, 'UNITS', units),
        patch.object(deploy, 'verify_runtime'),
        patch.object(
            deploy, 'restart', side_effect=[RuntimeError('startup'), None] if restart_fails else None
        ) as restart,
        patch.object(deploy, 'ready', side_effect=[True] if restart_fails else [False, True]),
    ):
        with pytest.raises(ValueError, match='previous release restored'):
            deploy.deploy(payload)
        assert (root / 'current').resolve() == previous
        assert restart.call_count == 2
        assert not list(units.rglob('20-ai-release.conf'))


def test_existing_commit_cannot_be_overwritten(tmp_path, payload):
    releases = tmp_path / 'releases'
    previous = releases / ('b' * 40)
    previous.mkdir(parents=True)
    (tmp_path / 'current').symlink_to(previous)
    target = releases / ('a' * 40)
    target.mkdir()
    (target / 'serve.py').write_text('different source')
    with patch.object(deploy, 'ROOT', tmp_path), patch.object(deploy, 'verify_runtime'):
        with pytest.raises(ValueError, match='different release contents'):
            deploy.deploy(payload)
        assert (target / 'serve.py').read_text() == 'different source'


def test_installed_versions_are_checked_even_with_matching_marker(tmp_path):
    lock = b'httpx==0.28.1\n'
    (tmp_path / 'ai-requirements.sha256').write_text(hashlib.sha256(lock).hexdigest())
    with (
        patch.object(deploy, 'CONFIG', tmp_path),
        patch.object(deploy, 'run', side_effect=subprocess.CalledProcessError(1, ['python'])) as run,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            deploy.verify_runtime({'requirements-cpu.lock': lock})
        assert json.loads(run.call_args.kwargs['input']) == {'httpx': '0.28.1'}


def test_installed_model_bundle_is_verified_before_activation(tmp_path):
    lock = b'httpx==0.28.1\n'
    (tmp_path / 'ai-requirements.sha256').write_text(hashlib.sha256(lock).hexdigest())
    primary, general = tmp_path / 'waste.pt', tmp_path / 'general.pt'
    primary.write_bytes(b'waste')
    general.write_bytes(b'general')
    manifest = {
        'filename': primary.name,
        'sha256': hashlib.sha256(primary.read_bytes()).hexdigest(),
        'general_model': {'filename': general.name, 'sha256': hashlib.sha256(general.read_bytes()).hexdigest()},
    }
    files = {'requirements-cpu.lock': lock, 'model.json': json.dumps(manifest).encode()}
    with (
        patch.object(deploy, 'CONFIG', tmp_path),
        patch.object(deploy, 'run'),
        patch.object(deploy, 'environment', return_value={'AI_MODEL_DIRECTORY': str(tmp_path)}),
    ):
        deploy.verify_runtime(files)
        general.write_bytes(b'changed')
        with pytest.raises(ValueError, match='Installed model'):
            deploy.verify_runtime(files)
        manifest['general_model']['filename'] = '../outside.pt'
        files['model.json'] = json.dumps(manifest).encode()
        with pytest.raises(ValueError, match='Invalid model filename'):
            deploy.verify_runtime(files)


@pytest.mark.parametrize('activation_succeeds', [True, False])
@pytest.mark.parametrize('was_active', [True, False])
def test_installed_demo_follows_release_manifest_and_restores_prior_command_on_failure(
    tmp_path, payload, activation_succeeds, was_active
):
    root = tmp_path / 'ai'
    previous = root / 'releases' / ('b' * 40)
    previous.mkdir(parents=True)
    (root / 'current').symlink_to(previous)
    units = tmp_path / 'units'
    units.mkdir()
    (units / 'wastenot-demo.service').write_text('[Service]\nExecStart=old-demo-command\n')
    override = units / 'wastenot-demo.service.d' / '20-ai-release.conf'
    override.parent.mkdir()
    original = b'[Service]\nExecStart=old-model-command\n'
    override.write_bytes(original)
    readiness = [True] if activation_succeeds else [False, True]
    with (
        patch.object(deploy, 'ROOT', root),
        patch.object(deploy, 'UNITS', units),
        patch.object(deploy, 'verify_runtime'),
        patch.object(deploy, 'restart') as restart,
        patch.object(deploy, 'demo_active', return_value=was_active),
        patch.object(deploy, 'ready', side_effect=readiness) as ready,
    ):
        if activation_succeeds:
            deploy.deploy(payload)
            command = override.read_text()
            assert f'ExecStart={root}/venv/bin/python {root}/current/serve.py demo' in command
            assert f'Environment=AI_MODEL_DIRECTORY={root}/models' in command
        else:
            with pytest.raises(ValueError, match='previous release restored'):
                deploy.deploy(payload)
            assert override.read_bytes() == original
            assert (root / 'current').resolve() == previous
        assert all(call.kwargs['include_demo'] == was_active for call in restart.call_args_list)
        assert all(call.kwargs['include_demo'] == was_active for call in ready.call_args_list)


@pytest.mark.parametrize(
    'loaded_model,verify_model,expected',
    [('combined-current', True, True), ('old-baseline', True, False), ('old-baseline', False, True)],
)
def test_demo_readiness_rejects_a_stale_model_and_accepts_legacy_recovery(
    tmp_path, loaded_model, verify_model, expected
):
    current = tmp_path / 'current'
    current.mkdir()
    (current / 'model.json').write_text(json.dumps({'model_version': 'combined-current'}))
    page = io.BytesIO(b'<meta name="wastenot-demo" content="test-session-nonce">')
    config = io.BytesIO(json.dumps({'model_version': loaded_model}).encode())
    opener = patch.object(urllib.request.OpenerDirector, 'open')
    with patch.object(deploy, 'ROOT', tmp_path), opener as open_request:
        open_request.side_effect = [page, config]
        assert deploy.demo_ready(urllib.request.build_opener(), verify_model=verify_model) == expected
