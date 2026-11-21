"""Offline acceptance and integrity checks for the delivered MAC-17 archive."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from analyze import header_verify, pool_summary, rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.root/'SHA256.json').read_text())
    for entry in manifest['files']:
        path=a.root/entry['path']; assert path.is_file(),path
        assert path.stat().st_size==entry['bytes'],path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],path
    source=header_verify(a.root/'sources/DeepSeek-V4-Flash')
    assert source['source_bytes']==159609485896 and source['main_layer_ids']==list(range(43))
    smoke=a.root/'remote/preflight-b021/results'
    for rank in range(8):
        record=json.loads((smoke/f'smoke-rank{rank}.json').read_text())
        assert record['integer_matmul_exact'] and record['all_reduce_correct'],record
    for name,expected in [('dp8-b021',[8192,32768,100000,200000]),
                          ('1m-b021',[1000000]),('1m-audit-b021',[1000000])]:
        result=json.loads((a.root/f'remote/{name}/results/result.json').read_text())
        assert result['exit_code']==0 and result['engine_started']
        assert [r['actual_input_tokens'] for r in result['requests']]==expected
        assert all(r['passed'] and r['actual_output_tokens']>=32 for r in result['requests'])
    actual=a.root/'remote/pool32b-b021/results'
    result=json.loads((actual/'result.json').read_text())
    assert result['exit_code']==0 and len(result['requests'])==16
    assert all(r['passed'] and r['actual_input_tokens']==32768 and r['actual_output_tokens']==32 for r in result['requests'])
    for rank in range(8):
        for kind in ['main','draft']:
            tensors=json.loads((actual/f'loaded-rank{rank}-{kind}.json').read_text())
            assert tensors['cpu_parameter_logical_bytes']==0
            if kind=='main':
                layers={int(m[1]) for t in tensors['tensors'] if (m:=re.match(r'model.layers.(\d+)\.',t['name']))}
                assert layers==set(range(43))
        stages={r['stage']:r for r in rows(actual/f'audit-rank{rank}.jsonl')}
        assert stages['after_load']['main_decoder_layer_count']==43
        assert len(stages['after_kv']['kv_config']['kv_cache_groups'])==6
    pool=pool_summary(actual/'pool-bytes.jsonl')
    assert pool['byte_validation_complete'] and pool['registered_workers']==8
    groups={re.search(r'@group:(\d+)@',r['key'])[1] for r in rows(actual/'pool-bytes.jsonl') if r['operation']=='get'}
    assert groups==set(map(str,range(6)))
    assert (a.root/'cleanup.json').exists()
    cleanup=json.loads((a.root/'cleanup.json').read_text())
    assert cleanup['own_containers_remaining']==[] and cleanup['own_ports_listening']==[]
    assert cleanup['reservation_released'] and cleanup['all_eight_npus_idle']
    print(json.dumps({'integrity_files':len(manifest['files']),'source_bytes':source['source_bytes'],
                      'all_lengths_passed':True,'pool_byte_validation':pool},indent=2))


if __name__=='__main__':main()
