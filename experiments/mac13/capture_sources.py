"""Capture public model metadata only; never fetch trained tensor shards."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--revision', help='Pin the official model revision')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
model_id = 'deepseek-ai/DeepSeek-V4.1-Flash'

def capture(name, url):
    entry = {'name': name, 'url': url,
             'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise ValueError('Metadata exceeds the 8 MiB limit')
            entry.update(status=response.status, bytes=len(body),
                         sha256=hashlib.sha256(body).hexdigest())
            (args.out / name).write_bytes(body)
    except urllib.error.HTTPError as exc:
        entry.update(status=exc.code, error='HTTPError')
    except Exception as exc:
        entry.update(status=None, error=type(exc).__name__)
    return entry

metadata_url = f'https://huggingface.co/api/models/{model_id}'
if args.revision:
    metadata_url += '/revision/' + args.revision
entries = [capture('hf-model.json', metadata_url)]
if entries[0]['status'] == 200:
    meta = json.loads((args.out / 'hf-model.json').read_text())
    revision = args.revision or meta['sha']
    urls = [
        ('config.json', f'https://huggingface.co/{model_id}/resolve/{revision}/config.json'),
        ('model.safetensors.index.json', f'https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors.index.json'),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        entries.extend(pool.map(lambda item: capture(*item), urls))
    (args.out / 'model-identity.json').write_text(json.dumps({
        'model_id': model_id, 'revision': revision,
        'architecture': meta.get('config', {}).get('architectures'),
        'safetensors': meta.get('safetensors'),
        'weight_downloaded': False,
    }, indent=2))
(args.out / 'source-manifest.json').write_text(json.dumps(entries, indent=2))
print(json.dumps(entries, indent=2))
