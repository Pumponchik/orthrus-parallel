"""Dependency-free verification shared by loading and downloading."""
import hashlib
from pathlib import Path


def checked_path(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe manifest path: {name}')
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError(f'Unsafe manifest path: {name}')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, item):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != item['size']:
        raise ValueError(f'Unexpected file size: {path}')
    sha = hashlib.sha256()
    git = hashlib.sha1(('blob ' + str(item['size']) + '\0').encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(block)
            git.update(block)
    expected = item.get('lfs', {}).get('oid')
    if (sha.hexdigest() if expected else git.hexdigest()) != (expected or item['oid']):
        raise ValueError(f'Hash mismatch: {path}')
    return sha.hexdigest()
