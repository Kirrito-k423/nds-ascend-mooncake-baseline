"""Fetch pinned public W8A8 metadata, with file-list hash verification."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--revision', required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
base = 'https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8/repo'
listing_url = base + '/files?' + urllib.parse.urlencode({'Revision': args.revision, 'Recursive': 'true'})
with urllib.request.urlopen(listing_url, timeout=30) as response:
    listing = response.read(2 * 1024 * 1024 + 1)
if len(listing) > 2 * 1024 * 1024:
    raise ValueError('File list exceeds limit')
(args.out / 'file-list.json').write_bytes(listing)
catalog = {f['Path']: f for f in json.loads(listing)['Data']['Files']}
names = ['config.json', 'aurora_export_manifest.json', 'quant_model_description.json',
         'quant_model_weights.safetensors.index.json', 'tokenizer_config.json', 'README.md']

def capture(name):
    if not name.endswith(('.json', '.md')):
        raise ValueError('Only allowlisted metadata files may be downloaded')
    url = base + '?' + urllib.parse.urlencode({'Revision': args.revision, 'FilePath': name})
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read(20 * 1024 * 1024 + 1)
    if len(body) > 20 * 1024 * 1024:
        raise ValueError('Metadata file exceeds limit')
    digest = hashlib.sha256(body).hexdigest()
    if digest != catalog[name]['Sha256']:
        raise ValueError('Metadata does not match pinned file list')
    (args.out / name).write_bytes(body)
    return {'name': name, 'url': url, 'revision': args.revision,
            'bytes': len(body), 'sha256': digest,
            'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(capture, names))
(args.out / 'source-manifest.json').write_text(json.dumps(records, indent=2))
print(json.dumps({'revision': args.revision, 'metadata_files': len(records),
                  'all_file_list_hashes_match': True,
                  'trained_weights_downloaded': False}, indent=2))
