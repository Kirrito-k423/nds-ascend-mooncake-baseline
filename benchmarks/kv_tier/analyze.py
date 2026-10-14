"""Validate recorded tier evidence before emitting request-level summaries."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import re


def percentile(values,q):
    values=sorted(values)
    if not values:return None
    p=(len(values)-1)*q;low=math.floor(p);high=math.ceil(p)
    return values[low]+(values[high]-values[low])*(p-low)


def io_bytes(text,kind='rbytes'):
    total=0
    for line in text.splitlines():
        for field in line.split()[1:]:
            if field.startswith(kind+'='):total+=int(field.split('=',1)[1])
    return total


def metric(text,name):
    values=[float(m.group(1)) for line in text.splitlines()
            if (m:=re.match(re.escape(name)+r'(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line))]
    assert values, 'Required runtime metric missing: '+name
    return sum(values)


def validate(evidence,requests,source_host):
    tier=evidence['tier'];length=evidence['input_tokens'];count=evidence['requests']
    assert requests and len(requests)==count
    assert all(r['success'] and r['usage']['prompt_tokens']==length and r['usage']['completion_tokens']==32
               and r['token_timing_exact'] for r in requests), 'Invalid request/token timing'
    loads=evidence['loads'];ends=[r for r in evidence['traces'] if r['op']=='get_end']
    role=evidence['role'];before_metrics=evidence['before'][role]['metrics'];after_metrics=evidence['after'][role]['metrics']
    computed=metric(after_metrics,'vllm:request_prefill_kv_computed_tokens_sum')-metric(before_metrics,'vllm:request_prefill_kv_computed_tokens_sum')
    local_hits=metric(after_metrics,'vllm:prefix_cache_hits_total')-metric(before_metrics,'vllm:prefix_cache_hits_total')
    assert local_hits==0, 'NPU cache hit contaminates external-tier comparison'
    if tier=='T0':
        assert not loads and not ends, 'Cold request restored external KV'
        assert computed==count*length, 'Cold prefill token count mismatch'
        return 0
    assert computed==count, 'External restore did not leave exactly one token to compute'
    assert len(loads)==count and all(int(local)==0 and int(external)==length-1 for _,local,external in loads)
    replicas=evidence['before_replicas']
    read_keys={k for row in ends for k in row['keys']}
    assert read_keys and read_keys.issubset(replicas), 'Loaded keys were not audited before reading'
    for key in read_keys:
        rows=replicas[key]
        assert rows and all(r['status'].rsplit('.',1)[-1]=='COMPLETE' for r in rows), 'Incomplete replica'
        if tier in ['T1','T2']:
            assert all(r['memory'] and source_host in json.dumps(r['memory_descriptor']) for r in rows), 'DRAM owner/mode mismatch'
        else:
            assert all(not r['memory'] and r['local_disk'] for r in rows), 'SSD has DRAM or other replicas'
    assert ends and all(all(value>0 for value in r['returns']) for r in ends), 'Native load failed'
    delivered=sum(sum(r['returns']) for r in ends)
    if tier in ['T3','T4']:
        assert sum(r['offload_rpc_after']-r['offload_rpc_before'] for r in ends)>0, 'No SSD RPC reads'
        before=evidence['before']['source'];after=evidence['after']['source']
        disk=sum(after['disk'][d]['read_bytes']-before['disk'][d]['read_bytes'] for d in before['disk'])
        assert disk>=delivered, 'No physical reads covering loaded bytes'
    return delivered


def main(root,allow_partial=False):
    root=Path(root);run=json.loads((root/'five-tier.json').read_text())
    complete=bool(run.get('sampling_finished') and not run.get('error'))
    assert complete or allow_partial, 'Experiment incomplete; use --allow-partial only for independently validated groups'
    node=json.loads((root/'node.json').read_text());source=node['source']
    rows=[]
    for phase in run['phases']:
        if phase['tier'] is None:continue
        label=phase['label']
        e=json.loads((root/(label+'.evidence.json')).read_text())
        requests=json.loads((root/(label+'.requests.json')).read_text())
        delivered=validate(e,requests,source)
        before=e['before']['source'];after=e['after']['source']
        container_read=(io_bytes(after['container_io_stat'])-io_bytes(before['container_io_stat'])
                        if 'container_io_stat' in before and 'container_io_stat' in after else None)
        for index,r in enumerate(requests):
            rows.append({'tier':e['tier'],'mode':label.split('-')[0],'role':e['role'],
                         'input_tokens':e['input_tokens'],'concurrency':e['concurrency'],'round':e['round'],
                         'request':index,'ttft_ms':r['ttft_ms'],'tpot_ms':r['tpot_ms'],
                         'batch_s':e['batch_s'],'batch_loaded_kv_bytes':delivered,
                         'batch_recomputed_tokens':metric(e['after'][e['role']]['metrics'],'vllm:request_prefill_kv_computed_tokens_sum')-metric(e['before'][e['role']]['metrics'],'vllm:request_prefill_kv_computed_tokens_sum'),
                         'source_container_read_bytes':container_read,'output_tokens':32})
    assert rows,'No formal requests'
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ['mode','tier','role','input_tokens','concurrency'])
        groups.setdefault(key,[]).append(row)
    summaries=[]
    for key,items in groups.items():
        assert {r['round'] for r in items}=={0,1,2}, 'Less than three complete rounds'
        item=dict(zip(['mode','tier','role','input_tokens','concurrency'],key))
        item.update(samples=len(items),tail_note='Descriptive only; <100 requests' if len(items)<100 else '')
        for field in ['ttft_ms','tpot_ms']:
            values=[r[field] for r in items]
            item.update({field+'_mean':statistics.mean(values),field+'_p50':percentile(values,.5),
                         field+'_p95':percentile(values,.95),field+'_p99':percentile(values,.99)})
        # Each batch duration appears once per request; deduplicate round windows.
        windows={r['round']:r['batch_s'] for r in items}
        item['output_tokens_per_s']=len(items)*32/sum(windows.values())
        summaries.append(item)
    with (root/'requests.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    (root/'summary.json').write_text(json.dumps({'validated':True,'experiment_complete':complete,
        'scope':'complete formal groups only','run_error':run.get('error'),'phases':summaries,
        'host_dram_traffic_measured':False,'network_link_bytes_attributed':False},indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main(sys.argv[1],allow_partial='--allow-partial' in sys.argv[2:])
