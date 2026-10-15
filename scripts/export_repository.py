"""Build a reviewable source archive from an explicit allowlist; never publish it."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('README.md', 'NOTICE.md', '.gitignore', '.python-version', 'pyproject.toml',
         'uv.lock', 'lab', 'bootstrap_uv.py')
DIRECTORIES = ('orthrus_parallel', 'scripts', 'tests', 'docs', 'configs', 'selectors', 'benchmarks', 'results')
EXTENSIONS = {'.py', '.md', '.json', '.csv', '.npz', '.svg', '.png', '.pdf'}


def repository_files():
    paths = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend(path for path in (ROOT / name).rglob('*')
                     if path.is_file() and path.suffix in EXTENSIONS and '__pycache__' not in path.parts)
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 5 * 1024**2:
            raise ValueError(f'Unexpected export item: {path}')
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new archive path')
    paths = list(repository_files())
    manifest = {str(p.relative_to(ROOT)): {'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
        for path in paths:
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.mode = 0o755 if path.name == 'lab' else 0o644
            with path.open('rb') as stream:
                archive.addfile(info, stream)
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('EXPORT_MANIFEST.json')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({'files':len(paths), 'bytes':args.output.stat().st_size,
        'archive':str(args.output), 'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
