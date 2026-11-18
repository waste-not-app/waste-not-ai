#!/usr/bin/env python3
"""Package only the worker source and its versioned runtime inputs."""
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package(output, commit):
    if not re.fullmatch('[a-f0-9]{40}', commit):
        raise ValueError('A full source commit is required')
    paths = [ROOT / name for name in ('serve.py', 'model.json', 'requirements-cpu.lock', 'contracts/backend-pin.json')]
    paths += sorted(path for path in (ROOT / 'src/wastenot_ai').rglob('*')
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
    files = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}
    files['release.json'] = (json.dumps({'commit': commit, 'sha256': {
        name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}, sort_keys=True) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz', format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            entry.mode = 0o444
            archive.addfile(entry, io.BytesIO(raw))
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print(package(Path(sys.argv[1]), commit))
