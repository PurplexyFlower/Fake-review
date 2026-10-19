"""Build an allowlisted deployment archive; preserve exact data/code bytes.

Only shell-script line endings are normalized for Linux. No .git, credentials,
old results, manuscript, private keys, or local virtual environment are shipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    'runs/train_run.py', 'runs/run_queue.py', 'runs/build_modern_dataset.py',
    'runs/build_source_grouped_datasets.py', 'runs/validate_source_grouped_plan.py',
    'runs/source_grouped_common.py', 'runs/capture_source_grouped_environment.py',
    'runs/evaluate_source_grouped.py', 'runs/package_source_grouped_results.py',
    'runs/aggregate_source_grouped_results.py', 'runs/ingest_source_grouped_packages.py',
    'runs/prepare_source_grouped_launch.py',
    'runs/check_source_grouped_gpu.py',
    'cloud/source_grouped_constraints.txt', 'cloud/setup_source_grouped.sh',
    'cloud/run_source_grouped_instance.sh', 'cloud/start_source_grouped_instance.sh',
    'cloud/SOURCE_GROUPED_RUNBOOK.md', 'tests/test_source_grouped_preflight.py',
    'cloud/source_grouped_instance_1.conf', 'cloud/source_grouped_instance_2.conf',
    'cloud/source_grouped_instance_3.conf',
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='dist/source_grouped_launch.tar.gz')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads((ROOT / 'launch_manifest.json').read_text(encoding='utf-8'))
        for relative, record in manifest['files'].items():
            data = (ROOT / relative).read_bytes()
            if len(data) != record['bytes'] or digest(data) != record['sha256']:
                raise RuntimeError(f'Deployed file differs from manifest: {relative}')
        print(f"Launch verified: {len(manifest['files'])} files, {manifest['run_count']} runs")
        return
    output = ROOT / args.output
    source_path = ROOT / 'dataset/source_grouped/manifest.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    for relative, expected in source['input_sha256'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise RuntimeError(f'Input data hash changed: {relative}')
    for record in source['artifacts'].values():
        if digest((ROOT / record['path']).read_bytes()) != record['sha256']:
            raise RuntimeError(f'Grouped data hash changed: {record["path"]}')

    paths = set(CODE) | set(source['input_sha256'])
    paths.add('dataset/source_grouped/manifest.json')
    paths.update(record['path'] for record in source['artifacts'].values())
    paths.update(f'grids/source_grouped_{kind}.txt'
                 for kind in ('directional', 'logo', 'rank_fixed_scale'))
    paths.update(f'grids/source_grouped/instance_{n}_{mode}.txt'
                 for n in (1, 2, 3) for mode in ('core', 'full'))
    payload = {}
    records = {}
    for relative in sorted(paths):
        original = (ROOT / relative).read_bytes()
        deployed = original.replace(b'\r\n', b'\n') if relative.endswith('.sh') else original
        payload[relative] = deployed
        records[relative] = {'bytes': len(deployed), 'sha256': digest(deployed),
                             'local_sha256': digest(original)}
    manifest = {'schema_version': 1, 'run_count': 27, 'shard_count': 3,
                'source_manifest_sha256': digest(source_path.read_bytes()),
                'files': records}
    payload['launch_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an earlier launch.
    with output.open('xb') as handle, tarfile.open(fileobj=handle, mode='w:gz') as archive:
        for relative, data in payload.items():
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o755 if relative.endswith('.sh') else 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(output, 'r:gz') as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member)
            assert handle is not None
            if handle.read() != payload[member.name]:
                raise RuntimeError(f'Archive verification failed: {member.name}')
    checksum = digest(output.read_bytes())
    output.with_suffix(output.suffix + '.sha256').write_text(
        f'{checksum}  {output.name}\n', encoding='utf-8')
    output.with_suffix(output.suffix + '.manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Archive: {output}')
    print(f'Files: {len(payload)}; bytes: {output.stat().st_size}; SHA-256: {checksum}')


if __name__ == '__main__':
    main()
