"""Install only the uv executable from a size/hash-verified public wheel.

Python 3.6 compatible for the CV10 host. No system packages or shell installers.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

root = Path(__file__).resolve().parent
tools = root / '.tools'
tools.mkdir(exist_ok=True)
if (tools / 'uv').exists():
    raise SystemExit('uv already exists; keeping the existing tool')
metadata = json.load(urllib.request.urlopen('https://pypi.org/pypi/uv/0.13.0/json', timeout=30))
choices = [x for x in metadata['urls'] if x['filename'].endswith('.whl')
           and 'manylinux_2_17_x86_64' in x['filename']]
assert len(choices) == 1, [x['filename'] for x in choices]
item = choices[0]
assert item['size'] < 100 * 1024**2
assert shutil.disk_usage(str(root)).free > 20 * 1024**3
part = tools / 'uv.whl.part'
h = hashlib.sha256()
with urllib.request.urlopen(item['url'], timeout=120) as remote, part.open('wb') as target:
    while True:
        block = remote.read(1024**2)
        if not block:
            break
        h.update(block)
        target.write(block)
assert part.stat().st_size == item['size']
assert h.hexdigest() == item['digests']['sha256']
with zipfile.ZipFile(str(part)) as archive:
    members = [x for x in archive.namelist() if x.endswith('/scripts/uv')]
    assert len(members) == 1, members
    (tools / 'uv').write_bytes(archive.read(members[0]))
os.chmod(str(tools / 'uv'), 0o755)
(root / 'uv_bootstrap_receipt.json').write_text(json.dumps({
    'version': metadata['info']['version'], 'wheel': item['filename'],
    'bytes': item['size'], 'sha256': item['digests']['sha256'],
    'url': item['url']}, indent=2))
part.unlink()
print('Installed uv', metadata['info']['version'], 'in', tools)
