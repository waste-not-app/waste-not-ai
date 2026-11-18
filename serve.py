#!/usr/bin/env python3
import argparse
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import re


def application(kind):
    release = Path(__file__).resolve().parent / 'release.json'
    if release.is_file():
        os.environ['WASTENOT_AI_RELEASE'] = json.loads(release.read_text())['commit']
    if kind in {'detection', 'demo'}:
        if kind == 'detection':
            for name in ('GEMINI_API_KEY', *(f'GEMINI_API_KEY_{i}' for i in range(1, 6)), 'AI_RECOMMENDATION_TOKEN'):
                os.environ.pop(name, None)
        else:
            os.environ.pop('AI_SERVICE_TOKEN', None)
        model_directory = os.getenv('AI_MODEL_DIRECTORY')
        general_model = None
        input_shape = 'adaptive'
        manifest = None
        if model_directory:
            manifest = json.loads((Path(__file__).resolve().parent / 'model.json').read_text('utf-8'))
            artifacts = [manifest]
            if manifest.get('general_model'):
                artifacts.append(manifest['general_model'])
            paths = []
            for artifact in artifacts:
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.pt', artifact['filename']):
                    raise ValueError('Invalid model filename in release manifest')
                path = Path(model_directory) / artifact['filename']
                if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                    raise ValueError('Model checksum does not match the release manifest')
                paths.append(path)
            model = paths[0]
            general_model = paths[1] if len(paths) == 2 else None
            input_shape = manifest.get('input_shape', 'adaptive')
        else:
            model = Path(os.environ['AI_MODEL_PATH'])
            if hashlib.sha256(model.read_bytes()).hexdigest() != os.environ['AI_MODEL_SHA256']:
                raise ValueError('Model checksum does not match the configured artifact')
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from PIL import Image
        from wastenot_ai.detector import load_detector
        from wastenot_ai.worker import create_app
        detector = load_detector(model, device='cpu', input_shape=input_shape, general_model_path=general_model)
        if manifest and detector.artifact['model_version'] != manifest['model_version']:
            raise ValueError('Loaded model identity differs from the release manifest')
        blank = io.BytesIO()
        Image.new('RGB', (64, 64)).save(blank, 'PNG')
        detector.detect(blank.getvalue())
        if kind == 'demo':
            from wastenot_ai.demo import create_demo_app
            return create_demo_app(detector=detector, demo_guides=True)
        return create_app(detector=detector)
    os.environ.pop('AI_SERVICE_TOKEN', None)
    from wastenot_ai.providers import GeminiProvider
    from wastenot_ai.recommendation import load_guides
    from wastenot_ai.recommendation_worker import create_recommendation_app
    guide_path = os.environ.get('AI_GUIDES_PATH')
    provider = GeminiProvider()
    # Leave time for the worker envelope before Android's 20-second read and Go's 23-second header deadline.
    if provider.timeout > 17:
        raise ValueError('GEMINI_TIMEOUT_SECONDS must not exceed 17 for the Android/Go API deadlines')
    return create_recommendation_app(provider=provider, timeout=provider.timeout + 1,
                                     guides=load_guides(Path(guide_path) if guide_path else None),
                                     demo_guides=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('service', choices=('detection', 'recommendation', 'demo'))
    args = parser.parse_args()
    import uvicorn
    backend = nullcontext()
    if args.service in {'detection', 'demo'}:
        import torch
        if torch.backends.cpu.get_cpu_capability() in {'DEFAULT', 'NO AVX'}:
            # The VM CPU cannot run NNPACK; avoid repeating unsupported-kernel initialization for every layer.
            backend = torch.backends.nnpack.flags(enabled=False)
    with backend:
        app = application(args.service)
        port = int(os.getenv('AI_PORT') or {'detection': 8091, 'recommendation': 8093, 'demo': 8092}[args.service])
        uvicorn.run(app, host=os.getenv('AI_BIND_HOST', '127.0.0.1'), port=port,
                    access_log=False, log_level='warning', workers=1)


if __name__ == '__main__':
    main()
