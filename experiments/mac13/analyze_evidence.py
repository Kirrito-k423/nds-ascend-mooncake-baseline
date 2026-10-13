"""Check checkpoint headers and report observed deployment boundaries."""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument('--evidence', type=Path)
source.add_argument('--archive', type=Path, help='Audit a delivery ZIP without extracting it')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root = zipfile.Path(args.archive, at='evidence/') if args.archive else args.evidence
out = args.out
out.mkdir(parents=True, exist_ok=True)
checkpoint = root / 'sources/w8a8'
index = json.loads((checkpoint / 'quant_model_weights.safetensors.index.json').read_text())
header_dir = checkpoint / 'all-headers'
header_files = (p for p in header_dir.iterdir() if p.name.endswith('.header.json'))
headers = {p.name.removesuffix('.header.json'): json.loads(p.read_text())
           for p in header_files}
assert set(headers) == set(index['weight_map'].values()), 'Incomplete header capture'
for name, shard in index['weight_map'].items():
    assert name in headers[shard], f'Index tensor missing: {name}'
dtype_bytes = collections.Counter()
category_bytes = collections.Counter()
width = {'I8': 1, 'F32': 4, 'BF16': 2}
tensor_count = 0
for header in headers.values():
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        tensor_count += 1
        size = tensor['data_offsets'][1] - tensor['data_offsets'][0]
        assert size == math.prod(tensor['shape']) * width[tensor['dtype']], name
        dtype_bytes[tensor['dtype']] += size
        if '.engram.embed.' in name:
            group = 'engram_embedding_and_scale'
        elif name.startswith('mtp.'):
            group = 'mtp'
        elif name.startswith(('vision.', 'aligner.')):
            group = 'vision_aligner'
        else:
            group = 'backbone_and_other'
        category_bytes[group] += size
total = sum(dtype_bytes.values())
assert total == index['metadata']['total_size'], 'Header bytes do not match index'
assert tensor_count == len(index['weight_map']), 'Unexpected header tensors'
description = json.loads((checkpoint / 'quant_model_description.json').read_text())
export = json.loads((checkpoint / 'aurora_export_manifest.json').read_text())
integrity = {}
for name in ['config.json', 'quant_model_description.json',
             'quant_model_weights.safetensors.index.json', 'README.md']:
    digest = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
    expected = export['files'][name]['sha256']
    integrity[name] = {'observed_sha256': digest, 'export_sha256': expected,
                       'matches_export': digest == expected}

smoke_dir = root / 'preflight-retry/retry-results'
final_dir = root / 'startup-final/results'
startup = json.loads((final_dir / 'startup-result.json').read_text())
operations = json.loads((root / 'operations.json').read_text())
assert operations['reservation_released'] and operations['machine_ready']
assert not operations['machine_task_ids'] and not operations['machine_reservation_ids']
assert all(r['archiveReady'] for r in operations['tasks'])
smoke = [json.loads((smoke_dir / f'smoke-rank{i}.json').read_text()) for i in range(8)]
assert all(r['integer_matmul_exact'] and r['all_reduce_correct'] for r in smoke)
assert startup['engine_started'] is False
assert startup['requests_attempted'] == 0
assert not startup['transformers_config_recognized']
assert not startup['target_architecture_registered']
assert startup['exit_code'] == 1

def parse_smi(path):
    text = path.read_text()
    rows = {}
    current = None
    for line in text.splitlines():
        match = re.match(r'^\|\s*(\d+)\s*\|\s*Ascend\S+\s*\|\s*(\w+)\s*\|', line)
        if match:
            current = int(match[1]); rows[current] = {'health': match[2]}
        elif current is not None and re.search(r'\|\s*NA\s*\|', line):
            used, capacity = re.findall(r'(\d+)\s*/\s*(\d+)', line)[-1]
            rows[current].update(used_mib=int(used), physical_mib=int(capacity))
    idle = {int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)\b', text)}
    assert set(rows) == set(range(8)) and idle == set(range(8))
    return rows

before = parse_smi(final_dir / 'npu-before.txt')
after = parse_smi(final_dir / 'npu-after.txt')
memory = []
for rank in range(8):
    b, a, s = before[rank], after[rank], smoke[rank]
    memory.append({'rank': rank, 'health': b['health'],
                   'physical_hbm_gib': b['physical_mib'] / 1024,
                   'before_used_mib': b['used_mib'],
                   'before_free_gib': (b['physical_mib'] - b['used_mib']) / 1024,
                   'after_used_mib': a['used_mib'],
                   'smoke_runtime_total_gib': s['total_hbm_bytes'] / 2**30,
                   'smoke_runtime_free_gib': s['free_hbm_bytes'] / 2**30,
                   'smoke_peak_allocated_bytes': s['smoke_peak_allocated_bytes'],
                   'target_weight_bytes': None, 'target_activation_peak_bytes': None,
                   'target_runtime_graph_workspace_bytes': None,
                   'target_kv_capacity_bytes': None, 'target_kv_token_capacity': None})
with (out / 'rank-memory.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(memory[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(memory)

free = sum(row['before_free_gib'] for row in memory)
summary = {
    'issue': 'MAC-13', 'machine_id': 'a5-b07-6',
    'official_model_revision': '5254448103b6254fc646dacbca1cbbf20e9216c1',
    'w8a8_revision': '155aced390c4692698400b744546a8bc396699c6',
    'image_id': (smoke_dir / 'image-id.txt').read_text().strip(),
    'versions': json.loads((final_dir / 'versions.json').read_text()),
    'python_runtime': json.loads((final_dir / 'python-runtime.json').read_text()),
    'smoke_passed_ranks': len(smoke),
    'startup': startup,
    'checkpoint_header_audit': {
        'shards': len(headers), 'tensors': tensor_count, 'tensor_bytes': total,
        'tensor_gib': total / 2**30, 'dtype_bytes': dict(dtype_bytes),
        'category_bytes': dict(category_bytes), 'trained_tensor_bytes_downloaded': 0,
        'export_metadata_integrity': integrity,
        'embedding_storage': description['optional']['embedding_storage'],
        'readme_describes_bf16': 'format: bf16' in (checkpoint / 'README.md').read_text(),
    },
    'source_layout_comparison_only': {
        'before_free_hbm_gib': free,
        'checkpoint_tensor_gib': total / 2**30,
        'ideal_equal_shard_tensor_gib': total / 8 / 2**30,
        'payload_exceeds_current_free_gib': total / 2**30 - free,
        'runtime_tensor_sharding_measured': False,
        'oom_observed': False,
    },
    'rank_memory': memory,
    'request_matrix': [{'prompt_tokens_requested': length, 'completion_tokens_requested': 32,
                        'requests_attempted': 0, 'success_rate': None,
                        'observed_prompt_tokens': None, 'observed_completion_tokens': None,
                        'reason': 'engine initialization failed before weight loading'}
                       for length in [8000, 32000, 100000, 200000]],
    'concurrency_handoff': {'target_config_available': False,
                            'reason': 'unsupported target architecture in the measured stack'},
    'cleanup': {'all_cards_idle': True,
                'reservation_released': operations['reservation_released'],
                'reservation_id': operations['reservation_id'],
                'reservation_released_at': operations['reservation_released_at']},
}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({'header_shards_validated': len(headers), 'tensor_count': tensor_count,
                  'smoke_passed_ranks': len(smoke), 'startup_exit_code': startup['exit_code'],
                  'source_tensor_gib': total / 2**30, 'current_free_hbm_gib': free,
                  'target_request_count': 0}, indent=2))
