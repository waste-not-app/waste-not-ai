#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse
import urllib.request


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: fetch_model.py destination-directory')
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'model.json').read_text())
    artifacts = [manifest]
    if manifest.get('general_model'):
        artifacts.append(manifest['general_model'])
    for artifact in artifacts:
        fetch(artifact, Path(sys.argv[1]))


def fetch(artifact, directory):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.pt', artifact['filename']):
        raise ValueError('Invalid model filename in release manifest')
    target = directory / artifact['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.part')
        try:
            headers = {}
            url = urlparse(artifact['url'])
            token = os.getenv('GITHUB_TOKEN')
            if token and url.scheme == 'https' and url.netloc == 'api.github.com':
                headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/octet-stream'}
            request = urllib.request.Request(artifact['url'], headers=headers)
            # GitHub redirects private release assets to a signed object URL.
            # Remove credentials when following a redirect to another origin.
            class AssetRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, request, response, code, message, headers, new_url):
                    redirected = super().redirect_request(request, response, code, message, headers, new_url)
                    previous_origin = urlparse(request.full_url)
                    next_origin = urlparse(new_url)
                    if redirected and (next_origin.scheme, next_origin.netloc) != (previous_origin.scheme, previous_origin.netloc):
                        redirected.remove_header('Authorization')
                    return redirected
            opener = urllib.request.build_opener(AssetRedirect())
            with opener.open(request, timeout=60) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != artifact['sha256']:
                raise ValueError('Model size or checksum invalid')
            temporary.write_bytes(raw)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(target.read_bytes()).hexdigest() != artifact['sha256']:
        raise ValueError('Existing model checksum differs; preserve it for inspection')
    print('Verified model', artifact['name'], artifact['sha256'])


if __name__ == '__main__':
    main()
