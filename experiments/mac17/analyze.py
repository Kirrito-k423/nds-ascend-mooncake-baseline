"""Recompute MAC-17 budgets from collected records, including legacy DP logs."""
import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def host_peaks(path):
    peaks = {}; samples = rows(path) if path.exists() else []
    for row in samples:
        device = None
        for line in row['stdout'].splitlines():
            match = re.match(r'\|\s*(\d+)\s*\|\s*Ascend950DT', line)
            if match: device = int(match[1])
            used = re.search(r'(\d+)\s*/\s*98304', line)
            if used and device is not None:
                peaks[device] = max(peaks.get(device, 0), int(used[1]))
    intervals=[(b['time_ns']-a['time_ns'])/1e9 for a,b in zip(samples,samples[1:])]
    return {'samples': len(samples), 'sleep_after_collection_s': 5,
            'observed_median_interval_s':statistics.median(intervals) if intervals else None,
            'observed_max_interval_s':max(intervals) if intervals else None, 'sampled_peak_mib': peaks,
            'first_time_ns': samples[0]['time_ns'] if samples else None,
            'last_time_ns': samples[-1]['time_ns'] if samples else None,
            'continuous_peak_observation': False}


def header_verify(source):
    index = json.loads((source/'model.safetensors.index.json').read_text())
    dtypes = {'BF16':2, 'F32':4, 'I64':8, 'I8':1, 'F8_E8M0':1, 'F8_E4M3':1}
    found = {}; counts = collections.Counter(); total = 0
    for path in (source/'headers').glob('*.header.json'):
        cursor = 0
        for name, tensor in sorted(json.loads(path.read_text()).items(),
                                   key=lambda kv: kv[1].get('data_offsets',[0])[0] if isinstance(kv[1],dict) else 0):
            if name == '__metadata__': continue
            assert name not in found
            assert index['weight_map'][name] == path.name.removesuffix('.header.json')
            first,last = tensor['data_offsets']
            size = math.prod(tensor['shape']) * dtypes[tensor['dtype']]
            assert first == cursor and last-first == size
            cursor = last; found[name] = tensor; total += size; counts[tensor['dtype']] += size
    assert set(found) == set(index['weight_map'])
    assert total == index['metadata']['total_size']
    return {'tensor_count':len(found), 'source_bytes':total, 'dtype_bytes':dict(counts),
            'main_layer_ids':sorted({int(m[1]) for name in found if (m:=re.search(r'^layers\.(\d+)\.',name))}),
            'mtp_tensor_count':sum(name.startswith('mtp.') for name in found),
            'tensor_payload_bytes_downloaded':0}


def budget(directory):
    rank_stages = collections.defaultdict(dict)
    for path in directory.glob('audit-rank*.jsonl'):
        for row in rows(path):
            # Old DP recordings all used local NPU ordinal 0 as filename/rank.
            rank = row['dp_rank'] if row.get('dp_rank') is not None else row['rank']
            assert row['stage'] not in rank_stages[rank], (directory,rank,row['stage'])
            rank_stages[rank][row['stage']] = row
    result = []
    for rank, stages in sorted(rank_stages.items()):
        loaded = stages.get('after_load'); profile = stages.get('after_profile'); kv = stages.get('after_kv')
        if not all((loaded,profile,kv)):
            result.append({'rank':rank,'incomplete_stages':list(stages)}); continue
        cfg = kv['kv_config']; banks = collections.Counter()
        for tensor in kv['layers'].values():
            if tensor['alias_of'] is None and tensor['reported_storage_bytes']:
                banks[tensor['reported_storage_bytes']] += 1
        groups = []
        for gid, group in enumerate(cfg['kv_cache_groups']):
            families = collections.Counter(json.dumps(s,sort_keys=True) for s in
                group['kv_cache_spec'].get('kv_cache_specs',{'single':group['kv_cache_spec']}).values())
            groups.append({'group_id':gid,'layer_count':len(group['layer_names']),
                'families':[{'count':n,'spec':json.loads(s)} for s,n in families.items()]})
        result.append({'rank':rank,'main_decoder_layer_count':loaded.get('main_decoder_layer_count'),
            'model_memory_usage_bytes':loaded['model_memory_usage'],
            'main_reported_storage_bytes':loaded['main_unique_reported_storage_bytes'],
            'draft_reported_storage_bytes':loaded['draft_unique_reported_storage_bytes'],
            'profile_activation_peak_bytes':profile['peak_activation_memory'],
            'profile_non_torch_bytes':profile['non_torch_memory'],
            'available_kv_bytes':profile['available_kv_cache_memory_bytes'],
            'kv_planned_bytes':sum(t['size'] for t in cfg['kv_cache_tensors']),
            'kv_unique_reported_storage_bytes':kv['unique_kv_reported_storage_bytes'],
            'kv_unreferenced_planned_bytes':sum(t['size'] for t in cfg['kv_cache_tensors'])-cfg['num_blocks']*(23*128*640+21*128*132),
            'kv_alignment_extra_storage_bytes':kv['unique_kv_reported_storage_bytes']-cfg['num_blocks']*(23*128*640+21*128*132),
            'kv_storage_banks':[{'bytes':size,'count':count} for size,count in sorted(banks.items())],
            'num_blocks':cfg['num_blocks'],'kv_groups':groups,'after_kv_memory':kv['memory'],
            'graph_workspace_separate_bytes':None})
    return result


def pool_summary(path):
    if not path.exists(): return {'attempted':False}
    data = rows(path); puts = [r for r in data if r['operation']=='put']; gets = [r for r in data if r['operation']=='get']
    return {'attempted':True,'registered_workers':sum(r['operation']=='register' for r in data),
        'put_keys':len(puts),'put_success_keys':sum(r['sdk_result']>=0 for r in puts),
        'get_keys':len(gets),'get_success_keys':sum(r['sdk_result']>=0 for r in gets),
        'matched_poisoned_get_keys':sum(r['matches_put'] is True and r['destination_poisoned'] for r in gets),
        'mismatched_get_keys':sum(r['matches_put'] is False for r in gets),
        'unverified_get_keys':sum(r['matches_put'] is None for r in gets),
        'put_transfer_bytes':sum(sum(r['sizes']) for r in puts if r['sdk_result']>=0),
        'get_transfer_bytes':sum(sum(r['sizes']) for r in gets if r['sdk_result']>=0),
        'byte_validation_complete':bool(gets) and all(r['sdk_result']>=0 and r['matches_put'] is True for r in gets)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); result={'source_layout':header_verify(a.evidence/'sources/DeepSeek-V4-Flash'), 'runs':{}}
    for directory in sorted((a.evidence/'remote').glob('*/results')):
        if not (directory/'result.json').exists(): continue
        name=directory.parent.name
        result['runs'][name]={'request_result':json.loads((directory/'result.json').read_text()),
                             'rank_budgets':budget(directory), 'host_hbm':host_peaks(directory/'host-hbm.jsonl'),
                             'pool':pool_summary(directory/'pool-bytes.jsonl')}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source_bytes':result['source_layout']['source_bytes'],
                      'runs':{k:len(v['rank_budgets']) for k,v in result['runs'].items()}}))


if __name__=='__main__': main()
