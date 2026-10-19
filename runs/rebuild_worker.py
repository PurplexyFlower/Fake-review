"""Bounded follow-up worker, strict status audit and exclusive final packaging."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

from rebuild_common import (OUT, PLAN, ROOT, adapter_args, complete, load_plan,
                            save_json, sha256)


def deployment_check():
    manifest_path = ROOT / 'rebuild_deployment_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('missing follow-up deployment identity')
    manifest = json.loads(manifest_path.read_text())
    if manifest['plan_sha256'] != sha256(PLAN):
        raise RuntimeError('deployed plan differs')
    for name, record in manifest['files'].items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError('unexpected deployment path')
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'deployed code differs: {name}')


def select_jobs(plan, shard, family):
    return [job for job in plan['jobs'] if job['shard'] == shard and
            (family == 'all' or (job['kind'] == 'tfidf') == (family == 'cpu'))]


def status(plan, shard):
    jobs = [job for job in plan['jobs'] if job['shard'] == shard]
    records = [{'id': job['id'], 'kind': job['kind'], 'complete': complete(plan, job)} for job in jobs]
    return {'shard': shard, 'gpu_completed': sum(r['complete'] and r['kind'] != 'tfidf' for r in records),
        'gpu_expected': 8, 'cpu_completed': sum(r['complete'] and r['kind'] == 'tfidf' for r in records),
        'cpu_expected': 8, 'jobs': records}


def worker(plan, shard, family):
    import fcntl
    control = OUT / 'control'
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / f'{family}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'another {family} worker already holds this checkout')
    python = ROOT / '.venv/bin/python'
    jobs = select_jobs(plan, shard, family)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4' if family == 'cpu' else '8',
                       MKL_NUM_THREADS='4' if family == 'cpu' else '8', CUDA_VISIBLE_DEVICES='0')
    # Warm the adapter kernels in a synthetic, non-reportable smoke check.
    if family == 'gpu':
        encoder_receipt = control / 'encoder_preflight.json'
        if not encoder_receipt.exists():
            subprocess.run([str(python), 'runs/check_rebuild_transformers.py'],
                cwd=ROOT, env=environment, check=True, pass_fds=(lock.fileno(),))
        receipt = control / f'gpu_preflight_{shard}.json'
        if not receipt.exists():
            command = [str(python), 'runs/check_source_grouped_gpu.py']
            subprocess.run(command, cwd=ROOT, env=environment, check=True, pass_fds=(lock.fileno(),))
            import shutil
            shutil.copy2(ROOT / 'results/source_grouped_preflight/receipt.json', receipt)
    for job in jobs:
        deployment_check()
        load_plan()
        if complete(plan, job):
            print('SKIP_VERIFIED', job['id'], flush=True)
            continue
        started = datetime.now(timezone.utc).isoformat()
        save_json(control / f'shard_{shard}_{family}_progress.json',
            {'job': job['id'], 'stage': 'running', 'started_at_utc': started})
        if job['kind'] == 'adapter':
            # New tags and the exact new data hash prevent reuse of an old run.
            from source_grouped_common import find_completed_run
            line = shlex.join(adapter_args(job))
            try:
                run_dir = find_completed_run(line)
                print('REUSE_EXACT_TRAINING', job['id'], run_dir.name, flush=True)
            except RuntimeError as error:
                if not str(error).startswith('no complete exact run found for '):
                    raise
                subprocess.run([str(python), 'runs/train_run.py', *adapter_args(job)],
                    cwd=ROOT, env=environment, check=True, pass_fds=(lock.fileno(),))
                run_dir = find_completed_run(line)
            command = [str(python), 'runs/rebuild_baseline.py', '--job-id', job['id'],
                       '--adapter-run-dir', str(run_dir.relative_to(ROOT))]
        else:
            command = [str(python), 'runs/rebuild_baseline.py', '--job-id', job['id']]
        print('START', job['id'], started, flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True, pass_fds=(lock.fileno(),))
        if not complete(plan, job):
            raise RuntimeError(f'job returned without passing all evidence gates: {job["id"]}')
        print('VERIFIED', job['id'], flush=True)
    save_json(control / f'shard_{shard}_{family}_progress.json',
        {'stage': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
         'job_ids': [job['id'] for job in jobs]})
    print('WORKER_COMPLETE', shard, family, flush=True)


def package(plan, shard):
    check = status(plan, shard)
    if check['gpu_completed'] != 8 or check['cpu_completed'] != 8:
        raise RuntimeError('all 8 GPU and 8 CPU jobs must pass before final packaging')
    from rebuild_baseline import environment
    save_json(OUT / f'shard_{shard}_environment.json', environment())
    paths = {PLAN, ROOT / 'rebuild_deployment_manifest.json',
             ROOT / 'dataset/source_grouped/manifest.json', OUT / f'shard_{shard}_environment.json'}
    paths.update(ROOT / name for name in plan['data_sha256'])
    deployed = json.loads((ROOT / 'rebuild_deployment_manifest.json').read_text())
    paths.update(ROOT / name for name in deployed['files'])
    paths.update(ROOT / name for name in ('runs/train_run.py', 'runs/source_grouped_common.py',
        'runs/check_source_grouped_gpu.py', 'cloud/source_grouped_constraints.txt'))
    for job in plan['jobs']:
        if job['shard'] == shard:
            paths.update(path for path in (OUT / job['id']).rglob('*') if path.is_file())
    paths.update(path for path in (OUT / 'control').glob('*') if path.is_file() and path.suffix != '.lock')
    manifest = {'plan_sha256': sha256(PLAN), 'shard': shard, 'strict_counts': check,
        'files': {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                  for path in sorted(paths)}}
    manifest_path = OUT / f'shard_{shard}_package_manifest.json'
    save_json(manifest_path, manifest, exclusive=True)
    paths.add(manifest_path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = OUT / f'followup_shard_{shard}_{stamp}.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, strict_timestamps=False, allowZip64=True) as handle:
        for path in sorted(paths):
            handle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise RuntimeError('archive CRC verification failed')
        for name, entry in manifest['files'].items():
            import hashlib
            with handle.open(name) as payload:
                digest = hashlib.file_digest(payload, 'sha256').hexdigest()
            if digest != entry['sha256'] or handle.getinfo(name).file_size != entry['bytes']:
                raise RuntimeError(f'archive payload verification failed: {name}')
    checksum = sha256(archive)
    with archive.with_suffix('.zip.sha256').open('x', encoding='utf-8') as handle:
        handle.write(f'{checksum}  {archive.name}\n')
    print('PACKAGE_COMPLETE', archive, checksum, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('validate', 'status', 'worker', 'package'))
    parser.add_argument('--shard', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--family', choices=('gpu', 'cpu', 'all'), default='gpu')
    args = parser.parse_args()
    plan = load_plan()
    if args.action == 'validate':
        deployment_check()
        print('FOLLOWUP_DEPLOYMENT_VALID', 'GPU=24 CPU=24')
    elif args.action == 'status':
        print(json.dumps(status(plan, args.shard)))
    elif args.action == 'worker':
        if args.family == 'all':
            raise RuntimeError('use separate gpu and cpu workers to avoid idle GPU billing')
        worker(plan, args.shard, args.family)
    else:
        package(plan, args.shard)


if __name__ == '__main__':
    main()
