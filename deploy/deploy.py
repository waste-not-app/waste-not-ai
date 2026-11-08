#!/usr/bin/env python3
"""Verify a bounded worker release, activate it and restore the prior source on failure."""
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path('/opt/wastenot-ai')
CONFIG = Path('/etc/wastenot')
UNITS = Path('/etc/systemd/system')
SERVICES = ('wastenot-yolo', 'wastenot-decisions')
MAX_ARCHIVE = 4 * 1024 * 1024
MAX_EXPANDED = 10 * 1024 * 1024


def parse_release(raw):
    if len(raw) > MAX_ARCHIVE:
        raise ValueError('Release archive exceeds limit')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
        expanded = compressed.read(MAX_EXPANDED + 1)
    if len(expanded) > MAX_EXPANDED:
        raise ValueError('Expanded release exceeds limit')
    files = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(expanded), mode='r:') as archive:
        for entry in archive:
            path = PurePosixPath(entry.name)
            if (len(files) >= 128 or not entry.isfile() or entry.name in files
                    or path.is_absolute() or '..' in path.parts or '\\' in entry.name
                    or str(path) != entry.name or not 0 < entry.size <= 2 * 1024 * 1024
                    or entry.pax_headers
                    or not (entry.name.startswith('src/wastenot_ai/') or entry.name in {
                        'serve.py', 'model.json', 'requirements-cpu.lock', 'contracts/backend-pin.json', 'release.json'})):
                raise ValueError('Unexpected release member')
            total += entry.size
            if total > 8 * 1024 * 1024:
                raise ValueError('Expanded release exceeds limit')
            files[entry.name] = archive.extractfile(entry).read()
    manifest = json.loads(files['release.json'])
    commit = manifest['commit']
    if not re.fullmatch('[a-f0-9]{40}', commit) or set(manifest['sha256']) != set(files) - {'release.json'}:
        raise ValueError('Invalid release manifest')
    required = {'serve.py', 'model.json', 'requirements-cpu.lock', 'contracts/backend-pin.json',
                'src/wastenot_ai/__init__.py', 'src/wastenot_ai/worker.py',
                'src/wastenot_ai/recommendation_worker.py'}
    if not required <= set(files):
        raise ValueError('Incomplete release')
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError('Release checksum mismatch')
    return commit, files


def run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, timeout=90, **kwargs)


def environment(name):
    path = CONFIG / name
    info = path.stat()
    if info.st_uid != 0 or info.st_mode & 0o077:
        raise ValueError('Worker environment must be root-owned and mode 0600')
    return dict(line.split('=', 1) for line in path.read_text().splitlines()
                if line and not line.startswith('#'))


def verify_runtime(files):
    lock = files['requirements-cpu.lock']
    if (CONFIG / 'ai-requirements.sha256').read_text().strip() != hashlib.sha256(lock).hexdigest():
        raise ValueError('Prepare the new locked CPU environment before deploying dependency changes')
    versions = dict(re.findall(r'^([a-zA-Z0-9_-]+)==([^\s]+)', lock.decode(), re.MULTILINE))
    # Inspect actual installed versions too: a matching marker cannot hide a changed environment.
    run(str(ROOT / 'venv/bin/python'), '-I', '-c',
        'import importlib.metadata as m,json,sys; expected=json.load(sys.stdin); '
        'assert sys.version_info[:2]==(3,12); '
        'assert all(m.version(name)==version for name,version in expected.items())',
        input=json.dumps(versions).encode())
    model = json.loads(files['model.json'])
    detection = environment('ai-detection.env')
    if detection.get('AI_MODEL_DIRECTORY'):
        artifacts = [model]
        if model.get('general_model'):
            artifacts.append(model['general_model'])
        for artifact in artifacts:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.pt', artifact['filename']):
                raise ValueError('Invalid model filename in release manifest')
            path = Path(detection['AI_MODEL_DIRECTORY']) / artifact['filename']
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                raise ValueError('Installed model differs from release manifest')
    elif (model.get('general_model') or detection['AI_MODEL_SHA256'] != model['sha256']
          or hashlib.sha256(Path(detection['AI_MODEL_PATH']).read_bytes()).hexdigest() != model['sha256']):
        raise ValueError('Installed model differs from release manifest')


def link(name, target):
    temporary = ROOT / ('.' + name + '-new')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    temporary.replace(ROOT / name)


def dropin(service):
    kind = {'wastenot-yolo': 'detection', 'wastenot-decisions': 'recommendation', 'wastenot-demo': 'demo'}[service]
    model_environment = f'Environment=AI_MODEL_DIRECTORY={ROOT}/models\n' if kind == 'demo' else ''
    return (f'[Service]\nWorkingDirectory={ROOT}/current\n'
            f'Environment=PYTHONPATH={ROOT}/current/src\n{model_environment}ExecStart=\n'
            f'ExecStart={ROOT}/venv/bin/python {ROOT}/current/serve.py {kind}\n').encode()


def restart():
    run('systemctl', 'daemon-reload')
    run('systemctl', 'restart', *SERVICES)
    try:
        run('systemctl', 'try-restart', 'wastenot-demo')
    except Exception:
        pass


def ready(commit):
    for _ in range(60):
        try:
            for port, name, key in ((8091, 'ai-detection.env', 'AI_SERVICE_TOKEN'),
                                    (8093, 'ai-recommendation.env', 'AI_RECOMMENDATION_TOKEN')):
                token = environment(name)[key]
                request = urllib.request.Request(f'http://127.0.0.1:{port}/health/ready',
                                                 headers={'Authorization': 'Bearer ' + token})
                with urllib.request.urlopen(request, timeout=2) as response:
                    value = json.load(response)
                    if not (value.get('ready') is True or value.get('data', {}).get('ready') is True):
                        raise ValueError('Worker is not ready')
                    if commit is not None and response.headers.get('X-WasteNot-Release') != commit:
                        raise ValueError('Worker is running a different source release')
            return True
        except (OSError, ValueError):
            time.sleep(1)
    return False


def deploy(raw):
    commit, files = parse_release(raw)
    verify_runtime(files)
    releases = ROOT / 'releases'
    releases.mkdir(exist_ok=True)
    target = releases / commit
    previous = (ROOT / 'current').resolve(strict=True) if (ROOT / 'current').is_symlink() else None
    if previous is None or previous.parent != releases or not re.fullmatch('[a-f0-9]{40}', previous.name):
        raise ValueError('An existing valid AI release is required; bootstrap separately')
    if target.exists():
        actual = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()}
        if actual != files:
            raise ValueError('Commit already has different release contents')
    else:
        stage = Path(tempfile.mkdtemp(prefix='.stage-', dir=releases))
        try:
            for name, content in files.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(0o444)
            for path in stage.rglob('*'):
                if path.is_dir():
                    path.chmod(0o755)
            stage.chmod(0o755)
            stage.replace(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    before = {}
    try:
        services = SERVICES + (('wastenot-demo',) if (UNITS / 'wastenot-demo.service').is_file() else ())
        for service in services:
            path = UNITS / (service + '.service.d') / '20-ai-release.conf'
            before[path] = path.read_bytes() if path.exists() else None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(dropin(service))
            path.chmod(0o644)
        link('current', target)
        restart()
        if not ready(commit):
            raise ValueError('New worker source did not become ready')
    except Exception:
        link('current', previous)
        for path, content in before.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        restart()
        if not ready(None):
            raise RuntimeError('Previous AI release also failed readiness') from None
        raise ValueError('AI deployment failed; previous release restored') from None
    if previous != target:
        link('previous', previous)
    return {'commit': commit, 'ready': True}


if __name__ == '__main__':
    try:
        if os.geteuid() != 0 or len(sys.argv) != 1:
            raise ValueError('Run the deployment helper as administrator')
        with open('/run/lock/wastenot-ai-deploy.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(deploy(sys.stdin.buffer.read(MAX_ARCHIVE + 1))))
    except Exception:
        print('AI deployment failed; inspect the release and worker state on the host.', file=sys.stderr)
        raise SystemExit(1)
