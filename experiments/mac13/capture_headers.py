"""Read safetensors headers via HTTP Range; never read tensor payloads."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import struct
import urllib.parse
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--revision', required=True)
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--workers', type=int, default=4)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
index = json.loads(args.index.read_text())
names = sorted(set(index['weight_map'].values()))

def read_range(url, first, last):
    req = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}'})
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 206:
            raise ValueError('Server did not honor metadata-only range request')
        content_range = response.headers.get('Content-Range', '')
        if not content_range.startswith(f'bytes {first}-{last}/'):
            raise ValueError('Unexpected Content-Range')
        result = response.read(last - first + 1)
        if len(result) != last - first + 1:
            raise ValueError('Incomplete header range')
        return result

def capture(name):
    url = 'https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8/repo?'
    url += urllib.parse.urlencode({'Revision': args.revision, 'FilePath': name})
    record = {'name': name, 'url': url, 'revision': args.revision,
              'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'tensor_payload_bytes_read': 0}
    try:
        prefix = read_range(url, 0, 7)
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= 4 * 1024 * 1024:
            raise ValueError('Invalid or oversized safetensors header')
        body = read_range(url, 8, 7 + size)
        parsed = json.loads(body)
        (args.out / (name + '.header.json')).write_bytes(body)
        record.update(header_bytes=size, sha256=hashlib.sha256(body).hexdigest(),
                      tensor_count=len([k for k in parsed if k != '__metadata__']),
                      tensor_bytes=sum(v['data_offsets'][1] - v['data_offsets'][0]
                                       for k, v in parsed.items() if k != '__metadata__'))
    except Exception as exc:
        record['error'] = type(exc).__name__
    return record

with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    records = list(pool.map(capture, names))
(args.out / 'range-manifest.json').write_text(json.dumps(records, indent=2))
summary = {'revision': args.revision, 'indexed_shards': len(names),
           'captured_shards': sum('error' not in r for r in records),
           'tensor_payload_bytes_read': 0,
           'header_bytes': sum(r.get('header_bytes', 0) for r in records),
           'tensor_bytes': sum(r.get('tensor_bytes', 0) for r in records),
           'index_tensor_bytes': index['metadata']['total_size']}
(args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if summary['captured_shards'] != len(names):
    raise SystemExit(1)
