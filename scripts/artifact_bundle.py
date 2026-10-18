"""Pack, verify and restore local benchmark artifacts with hashes and no overwrite."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from scripts.evaluate_visual_suite import source_hash
from utils.checkpoint import file_hash
from utils.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]
P5_FILES = ('data/grounding_p5_labels.h5', 'weights/p5_student/best.pt',
            'data/grounding_p5_labels.report.json', 'artifacts/p5_student/test/report.json',
            'artifacts/p5_student/val_refined/report.json', 'artifacts/p5_student/closed_loop_20.json',
            'artifacts/p5_student/cpu_throughput.json')


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError('Unsafe artifact path')
    if path.parts[0] not in ('data', 'weights', 'configs', 'artifacts'):
        raise ValueError('Unexpected artifact directory')
    return path


def pack(output, manifest_path):
    frozen = json.loads((ROOT/'docs/operations/REPRO_BASELINE.json').read_text())
    files = set(frozen['files_sha256']) | set(P5_FILES)
    # Verify the historical artifacts without replacing the frozen baseline.
    for path, expected in frozen['files_sha256'].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError('Frozen artifact hash mismatch: '+path)
    manifest = {'schema': 'omni-artifact-bundle-v1', 'source_sha256': source_hash(),
                'files': {path: {'sha256': file_hash(ROOT/path), 'size': (ROOT/path).stat().st_size}
                          for path in sorted(files)},
                'baseline_manifest_sha256': file_hash(ROOT/'docs/operations/REPRO_BASELINE.json')}
    output = Path(output)
    if output.exists() or Path(manifest_path).exists():
        raise FileExistsError('Bundle output or manifest already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        for path in manifest['files']:
            archive.write(ROOT/path, path)
    manifest['archive_sha256'] = file_hash(output)
    write_json(manifest_path, manifest)
    print(json.dumps({'archive': str(output), 'sha256': manifest['archive_sha256'], 'files': len(files)}))
    return manifest


def verify_restore(archive_path, destination=None, expected_sha256=None):
    if expected_sha256 and file_hash(archive_path) != expected_sha256:
        raise ValueError('Archive checksum mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or 'manifest.json' not in names:
            raise ValueError('Duplicate or missing archive manifest')
        if archive.getinfo('manifest.json').file_size > 1024*1024:
            raise ValueError('Manifest too large')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('schema') != 'omni-artifact-bundle-v1':
            raise ValueError('Unsupported bundle schema')
        files = manifest['files']
        if set(names) != set(files) | {'manifest.json'}:
            raise ValueError('Archive members differ from manifest')
        for name, metadata in files.items():
            safe_path(name)
            info = archive.getinfo(name)
            if stat.S_ISLNK(info.external_attr >> 16) or info.file_size != metadata['size']:
                raise ValueError('Symlink or size mismatch')
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != metadata['sha256']:
                raise ValueError('Artifact checksum mismatch: '+name)
        if destination is not None:
            root = Path(destination).resolve()
            if not root.is_dir():
                raise ValueError('Destination must be an existing checkout directory')
            # Preflight every target before writing anything; conflicting files
            # and escaping symlink parents are never replaced.
            for name, metadata in files.items():
                target = root/name
                if not target.resolve().is_relative_to(root):
                    raise ValueError('Destination symlink escapes checkout')
                if target.is_symlink() or (target.exists() and file_hash(target) != metadata['sha256']):
                    raise FileExistsError('Conflicting artifact: '+name)
            with tempfile.TemporaryDirectory(prefix='.artifact-stage-', dir=root) as staging:
                for name in files:
                    target = root/name
                    if target.exists():
                        continue
                    temporary = Path(staging)/name
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, temporary.open('xb') as sink:
                        shutil.copyfileobj(source, sink)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Exclusive final copy protects against a concurrent writer.
                    with temporary.open('rb') as source, target.open('xb') as sink:
                        shutil.copyfileobj(source, sink)
        print(json.dumps({'verified_files': len(files), 'restored': destination is not None}))
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pack', 'verify', 'restore'])
    p.add_argument('--archive', required=True)
    p.add_argument('--manifest', default='docs/operations/ARTIFACT_BUNDLE.json')
    p.add_argument('--destination', default='.')
    p.add_argument('--sha256', help='Trusted archive checksum, required for restore')
    args = p.parse_args()
    if args.action == 'pack':
        pack(args.archive, args.manifest)
    else:
        if args.action == 'restore' and not args.sha256:
            p.error('Restore requires --sha256 from the trusted bundle manifest')
        verify_restore(args.archive, args.destination if args.action == 'restore' else None, args.sha256)


if __name__ == '__main__':
    main()
