from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'SOURCE_367.json').read_text(encoding='utf-8'))
issues = []
for name, record in manifest['files'].items():
    path = root / name
    if not path.is_file() or path.stat().st_size != record['size']:
        issues.append(name)
        continue
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        issues.append(name)
if issues:
    raise SystemExit('Missing or different files: ' + ', '.join(issues))
print(manifest['release'], len(manifest['files']), 'files verified')
