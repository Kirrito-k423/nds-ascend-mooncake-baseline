"""Five physical KV tiers; collect requests, replicas, I/O and failures in foreground."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
import traceback

from node import Node, http, serve
from stream import request
from segments import contributing_segments

ROLE = sys.argv[1]
NODE = Node(ROLE)
OUT = NODE.out
REMOTE = 'http://'+os.environ['NDS_DESTINATION_HOST']+':18198'
BASE = {'source':'http://127.0.0.1:18196',
        'destination':'http://'+os.environ['NDS_DESTINATION_HOST']+':18196'}
REPORT = {'model':'Qwen2.5-72B-Instruct','dummy':True,'input_kind':'fixed synthetic token IDs',
          'output_tokens':32,'rounds':3,'phases':[],'probes':[],'skipped':[],
          'direct_path_available':None,'direct_path_status':'no validated callable SSD-to-HBM path',
          'host_dram_traffic_measured':False,
          'start_wall_ns':time.time_ns()}


def save():
    (OUT/'five-tier.json').write_text(json.dumps(REPORT,indent=2))
    print(json.dumps({'phases':len(REPORT['phases']),'probes':len(REPORT['probes']),
                      'last':REPORT['phases'][-1]['label'] if REPORT['phases'] else None,
                      'error':REPORT.get('error')}),flush=True)


def call(role, path, body=None, timeout=1800):
    body = {} if body is None else body
    if role == 'source': return NODE.dispatch(path,body)
    return json.loads(http(REMOTE,path,body,timeout))


def agent(path, body): return call('source','/agent',{'path':path,'body':body})

def disk_only(keys):
    text=call('source','/log',{'offset':0})['text']
    names=contributing_segments(text,NODE.source)
    return agent('/disk-only',{'keys':keys,'segments':names})


def traces(text):
    return [json.loads(line.split('NDS_TRACE ',1)[1]) for line in text.splitlines() if 'NDS_TRACE ' in line]


def keys_from(text):
    return sorted({k for row in traces(text) if row['op'] == 'put' for k in row['keys']})


def reset(role): http(BASE[role],'/reset_prefix_cache?reset_external=false',{},timeout=60)


def dataset(length, count, seed):
    rng = random.Random(seed)
    rows = [[rng.randrange(200,8000) for _ in range(length)] for _ in range(count)]
    fingerprints = [hashlib.sha256(json.dumps(x[:128]).encode()).hexdigest() for x in rows]
    assert len(set(fingerprints)) == count
    return rows


def batch(role, tokens, label, tier=None, round_number=None, before_replicas=None):
    reset(role)
    before = {r:call(r,'/snapshot') for r in ['source','destination']}
    offset = call(role,'/offset')['offset']
    observations = []
    done = threading.Event()
    def monitor():
        while not done.wait(.2):
            try: observations.append({'wall_ns':time.time_ns(),
                                      'metrics':http(BASE[role],'/metrics',timeout=2).decode()})
            except Exception as e: observations.append({'error':str(e)})
    sampler = threading.Thread(target=monitor,daemon=True);sampler.start()
    barrier = threading.Barrier(len(tokens))
    def one(ids):
        barrier.wait(timeout=10)
        return request(BASE[role],ids)
    begin = time.perf_counter_ns()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tokens)) as pool:
            results = list(pool.map(one,tokens))
    finally:
        done.set();sampler.join(timeout=5)
    elapsed = (time.perf_counter_ns()-begin)/1e9
    time.sleep(1) # ensure asynchronous cache saves are visible; excluded from latency.
    after = {r:call(r,'/snapshot') for r in ['source','destination']}
    text = call(role,'/log',{'offset':offset})['text']
    (OUT/(label+'.log')).write_text(text)
    (OUT/(label+'.requests.json')).write_text(json.dumps(results,indent=2))
    (OUT/(label+'.sampling.json')).write_text(json.dumps(observations))
    evidence = {'label':label,'role':role,'tier':tier,'round':round_number,
                'input_tokens':len(tokens[0]),'concurrency':len(tokens),'requests':len(results),
                'successful':sum(x['success'] for x in results),'batch_s':elapsed,
                'before':before,'after':after,'before_replicas':before_replicas,
                'traces':traces(text),'loads':re.findall(
                    r'KV pool load spec created req=(\S+) vllm_cached=(\d+) kvpool_cached=(\d+)',text),
                'token_timing_exact':all(x['token_timing_exact'] for x in results),
                'keys':keys_from(text)}
    (OUT/(label+'.evidence.json')).write_text(json.dumps(evidence,indent=2))
    REPORT['phases'].append({k:v for k,v in evidence.items() if k not in ['before','after','traces','before_replicas']})
    save()
    assert evidence['successful'] == len(results), 'Request failure: '+label
    assert evidence['token_timing_exact'], 'SSE token grouping prevents exact interval measurement: '+label
    if tier == 'T0': assert not evidence['loads'], 'Unexpected external hit: '+label
    elif tier:
        assert len(evidence['loads']) == len(results), 'Missing load spec: '+label
        assert all(int(local)==0 and int(ext)==len(tokens[0])-1 for _,local,ext in evidence['loads']), 'Partial/local hit: '+label
        ends = [x for x in evidence['traces'] if x['op']=='get_end']
        assert ends and all(all(n>0 for n in x['returns']) for x in ends), 'Native load failed'
        if tier in ['T3','T4']:
            assert sum(x['offload_rpc_after']-x['offload_rpc_before'] for x in ends)>0, 'No offload RPC reads'
            delivered = sum(sum(x['returns']) for x in ends)
            disk_read = sum(after['source']['disk'][d]['read_bytes']-before['source']['disk'][d]['read_bytes']
                            for d in before['source']['disk'])
            assert disk_read >= delivered, ('Disk counters do not cover restored bytes',disk_read,delivered)
    return evidence


def memory_audit(keys):
    evidence = agent('/audit',{'keys':keys})
    assert keys and all(rows and all(r['memory'] and r['status'].rsplit('.',1)[-1]=='COMPLETE' and
                         NODE.source in json.dumps(r['memory_descriptor']) for r in rows)
                        for rows in evidence.values()), 'Require source MEMORY-only replicas'
    return evidence


def wait_disk(keys):
    deadline = time.monotonic()+180
    while time.monotonic() < deadline:
        evidence = agent('/audit',{'keys':keys})
        if all(rows and any(r['local_disk'] and r['status'].rsplit('.',1)[-1]=='COMPLETE' for r in rows)
               for rows in evidence.values()): return evidence
        time.sleep(1)
    raise TimeoutError('Persistence did not complete in 180s')


def micro(mode):
    # The installed SSD reader aborts for 64MiB objects. W1 independently
    # validated 1/3MiB; MEMORY retains its original larger-object control.
    sizes=[1024**2,3*1024**2] if mode=='disk' else [1024**2,64*1024**2]
    for size in sizes:
        manifest = agent('/probe-put',{'size':size})
        key = manifest['key']
        if mode == 'disk': wait_disk([key])
        for round_number in range(3):
            for role in (['source','destination'] if round_number%2==0 else ['destination','source']):
                proof = (disk_only([key])['after'] if mode == 'disk'
                         else memory_audit([key]))
                before = {r:call(r,'/snapshot') for r in ['source','destination']}
                result = call(role,'/agent',{'path':'/probe-read','body':{'key':key,'size':size}})
                after = {r:call(r,'/snapshot') for r in ['source','destination']}
                assert result['sha256'] == manifest['sha256'] and result['exact']
                REPORT['probes'].append({'mode':mode,'role':role,'round':round_number,
                                         'manifest':manifest,'replicas':proof,'before':before,
                                         'read':result,'after':after})
                save()
    agent('/remove-all',{})


def capacity():
    text = call('source','/log',{'offset':0})['text']
    candidates = re.findall(r'(?:GPU|NPU) KV cache size:\s*([\d,]+) tokens',text)
    if not candidates: candidates = re.findall(r'KV cache.*?([\d,]+) tokens',text)
    if not candidates: return None
    return min(int(x.replace(',','')) for x in candidates)


def owner():
    try:
        assert NODE.local != os.environ['NDS_DESTINATION_HOST']
        deadline = time.monotonic()+300
        while time.monotonic()<deadline:
            try: json.loads(http(REMOTE,'/',timeout=2));break
            except OSError: time.sleep(2)
        else: raise TimeoutError('Destination controller unavailable')
        # First prove all five tiers at 1024/C1; launch the capacity scan only
        # after reviewing that complete foreground calibration result.
        configurations = json.loads(os.environ.get('NDS_CONFIGS_JSON','[[1024,1]]'))
        assert configurations and all(isinstance(v,list) and len(v)==2 and v[0] in [1024,8192,16384,24576]
                                      and v[1] in [1,8,25] for v in configurations)
        REPORT['configurations'] = configurations;save()
        modes = json.loads(os.environ.get('NDS_MODES_JSON','["memory","disk"]'))
        assert modes and all(x in ['memory','disk'] for x in modes) and len(set(modes))==len(modes)
        REPORT['modes']=modes;save()
        for mode in modes:
            call('source','/start',{'mode':mode})
            call('destination','/start',{'mode':mode})
            cap = capacity();REPORT[mode+'_kv_token_capacity'] = cap;save()
            # Native data validation precedes model timing.
            micro(mode)
            for c in sorted({c for _,c in configurations}):
                if cap and c*(1024+32) > .8*cap: continue
                # Each worker has its own TE endpoint. Agent-level probes do
                # not warm model-worker links; include actual external loads.
                for warm_round in range(3):
                    seed=801000+c*100+warm_round
                    common=dataset(1024,c,seed)
                    label=f'{mode}-warm-C{c}-W{warm_round}'
                    batch('destination',dataset(1024,c,seed+8000000),label+'-cold-dest')
                    batch('source',common,label+'-cold-source')
                    batch('source',common,label+'-local-load')
                    batch('destination',common,label+'-remote-load')
            for length,c in configurations:
                if cap is None and c != 1:
                    REPORT['skipped'].append({'mode':mode,'input':length,'concurrency':c,'reason':'KV capacity not parsed'})
                    save();continue
                if cap and c*(length+32) > .8*cap:
                    REPORT['skipped'].append({'mode':mode,'input':length,'concurrency':c,
                                              'reason':'Above 80% calibrated KV capacity','capacity':cap})
                    save();continue
                for rn in range(3):
                    label=f'{mode}-L{length}-C{c}-R{rn}'
                    seed = length*10000+c*100+rn
                    tokens = dataset(length,c,seed)
                    (OUT/(label+'.inputs.json')).write_text(json.dumps({'seed':seed,'tokens':tokens,
                        'first_block_hashes':[hashlib.sha256(json.dumps(x[:128]).encode()).hexdigest() for x in tokens]}))
                    agent('/remove-all',{})
                    # A matched cold destination baseline uses different prefixes.
                    batch('destination',dataset(length,c,seed+700000000),label+'-T0-dest','T0',rn)
                    agent('/remove-all',{})
                    cold = batch('source',tokens,label+'-T0-source','T0',rn)
                    prime = batch('source',tokens,label+'-prime')
                    keys = sorted(set(cold['keys']+prime['keys']))
                    assert keys, 'No recorded cache save keys'
                    for tier,role in ([('T1','source'),('T2','destination')] if mode=='memory' else
                                      [('T3','source'),('T4','destination')]):
                        reset('source');reset('destination')
                        if mode == 'memory': proof=memory_audit(keys)
                        else:
                            wait_disk(keys)
                            clearing=disk_only(keys)
                            (OUT/(label+'-'+tier+'.clearing.json')).write_text(json.dumps(clearing,indent=2))
                            proof=clearing['after']
                        batch(role,tokens,label+'-'+tier,tier,rn,proof)
            call('destination','/stop');call('source','/stop')
        REPORT['sampling_finished']=True
    except Exception:
        REPORT['error']=traceback.format_exc();raise
    finally:
        save()
        try: call('destination','/stop')
        except Exception: pass
        try: call('destination','/shutdown')
        except Exception: pass
        NODE.stop()
        NODE.cleanup_offload()
        REPORT['finish_wall_ns']=time.time_ns();save()


if ROLE == 'source': owner()
else: serve(NODE)
