"""Freeze the additional 24-run GPU plan; no remote or training actions."""
import argparse
import io
import json
import tarfile
from pathlib import Path
from rebuild_common import ROOT, PLAN, make_plan, save_json, sha256

CODE = ('runs/rebuild_common.py', 'runs/rebuild_baseline.py', 'runs/rebuild_worker.py',
        'runs/prepare_rebuild.py', 'runs/check_rebuild_transformers.py',
        'tests/test_rebuild.py', 'cloud/run_rebuild.sh')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default=None)
    args = parser.parse_args()
    plan = make_plan()
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != plan:
            raise RuntimeError('existing frozen follow-up plan differs; refusing overwrite')
    else:
        save_json(PLAN, plan, exclusive=True)
    if args.archive:
        payload = {}
        for name in (*CODE, PLAN.relative_to(ROOT).as_posix()):
            content = (ROOT / name).read_bytes()
            if name.endswith('.sh'):
                content = content.replace(b'\r\n', b'\n')
            payload[name] = content
        manifest = {'plan_sha256': sha256(PLAN), 'files': {
            name: {'bytes': len(content), 'sha256': __import__('hashlib').sha256(content).hexdigest()}
            for name, content in payload.items()}}
        payload['rebuild_deployment_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
        destination = ROOT / args.archive
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as archive:
            for name, content in payload.items():
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(content), 315532800, 0o644
                archive.addfile(info, io.BytesIO(content))
        print('FOLLOWUP_ARCHIVE', destination, sha256(destination))
    print('PLAN_FROZEN', sha256(PLAN), 'GPU=24 CPU=24 SHARDS=3')


if __name__ == '__main__':
    main()
