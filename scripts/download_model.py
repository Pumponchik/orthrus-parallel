"""Show the pinned public download plan; pass --download to execute it."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
from urllib.parse import urlparse, quote
import tempfile
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT))
from orthrus_parallel.integrity import checked_path, verify_file

verify = verify_file


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != 'https':
            raise ValueError('Refusing a non-HTTPS download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_file(url, path, item):
    """Verify a private temporary file before atomically publishing it."""
    if urlparse(url).scheme != 'https':
        raise ValueError('Downloads require HTTPS')
    path = Path(path)
    opener = urllib.request.build_opener(HTTPSRedirectHandler())
    part = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                prefix=path.name + '.', suffix='.part', delete=False) as target:
            part = Path(target.name)
            with opener.open(url, timeout=180) as response:
                if urlparse(response.geturl()).scheme != 'https':
                    raise ValueError('Downloads require HTTPS')
                written = 0
                while block := response.read(8 * 1024**2):
                    written += len(block)
                    if written > item['size']:
                        raise ValueError('Server sent more bytes than the pinned manifest allows')
                    target.write(block)
            target.flush()
            os.fsync(target.fileno())
        digest = verify(part, item)
        part.replace(path)
        return digest
    finally:
        if part is not None:
            part.unlink(missing_ok=True)


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
        path = checked_path(dest, item['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            digest = verify(path, item)
        else:
            url = f'https://huggingface.co/{config["repo"]}/resolve/{config["revision"]}/{quote(item["path"], safe="/")}'
            digest = download_file(url, path, item)
        receipts.append({'file': item['path'], 'bytes': item['size'], 'sha256': digest})
        print('Verified', item['path'], flush=True)
    (ROOT / 'checkpoint_receipt.json').write_text(json.dumps({'state':'complete',
        'repo':config['repo'], 'revision':config['revision'], 'files':receipts}, indent=2))


if __name__ == '__main__':
    main()

