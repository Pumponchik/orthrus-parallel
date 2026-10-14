"""Show the pinned public download plan; pass --download to execute it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        raise ValueError(f'Unexpected file size: {path}')
    sha = hashlib.sha256()
    git = hashlib.sha1(('blob ' + str(item['size']) + '\0').encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(block)
            git.update(block)
    expected = item.get('lfs', {}).get('oid')
    actual = sha.hexdigest() if expected else git.hexdigest()
    if actual != (expected or item['oid']):
        raise ValueError(f'Hash mismatch: {path}')
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/checkpoint.json').read_text())
    files = [f for f in config['files'] if f['type'] == 'file']
    size = sum(f['size'] for f in files)
    dest = ROOT / 'models' / config['repo'].split('/')[-1] / config['revision']
    free = shutil.disk_usage(ROOT).free
    print(json.dumps({'repo': config['repo'], 'revision': config['revision'],
        'bytes': size, 'free_bytes': free, 'destination': str(dest), 'download_requested': args.download}, indent=2))
    if not args.download:
        return
    if free < 20 * 1024**3 or size > 5 * 1024**3:
        raise RuntimeError('Download exceeds the configured size/free-space bounds')
    dest.mkdir(parents=True, exist_ok=True)
    receipts = []
    for item in files:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint manifest path')
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            digest = verify(path, item)
        else:
            part = path.with_suffix(path.suffix + '.part')
            if part.exists():
                raise FileExistsError(f'Inspect the existing temporary file before retrying: {part}')
            url = f'https://huggingface.co/{config["repo"]}/resolve/{config["revision"]}/{item["path"]}'
            try:
                with urllib.request.urlopen(url, timeout=180) as response, part.open('xb') as target:
                    written = 0
                    while block := response.read(8 * 1024**2):
                        written += len(block)
                        if written > item['size']:
                            raise ValueError('Server sent more bytes than the pinned manifest allows')
                        target.write(block)
                digest = verify(part, item)
                part.replace(path)
            finally:
                if part.exists():
                    part.unlink()  # Only the temporary file created by this operation.
        receipts.append({'file': item['path'], 'bytes': item['size'], 'sha256': digest})
        print('Verified', item['path'], flush=True)
    (ROOT / 'checkpoint_receipt.json').write_text(json.dumps({'state':'complete',
        'repo':config['repo'], 'revision':config['revision'], 'files':receipts}, indent=2))


if __name__ == '__main__':
    main()
