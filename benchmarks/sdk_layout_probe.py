"""Native SDK model-shaped KV readback, without a model or kernel changes.

Storage: 80 layers x K/V x blocks x (128 tokens x 2 KV heads x 128 BF16).
Each key gathers 160 separated 64KiB buffers: 10MiB/key, 80MiB/rank at K8.
The CPU parent never initializes a device. Spawned workers preserve SDK import
ordering and retain all failures. JSON control is for a task-owned coordinator.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

LAYERS=80
BUFFER_BYTES=128*2*128*2
BUFFERS=LAYERS*2
KEY_BYTES=BUFFERS*BUFFER_BYTES
PREFIX='MAC16_LAYOUT_RESULT '


def pointer_lists(base,keys):
    return [[base+(i*keys+k)*BUFFER_BYTES for i in range(BUFFERS)] for k in range(keys)]


def is_complete(status):return status.rsplit('.',1)[-1]=='COMPLETE'


def serial(obj,depth=0):
    if obj is None or isinstance(obj,(str,int,float,bool)):return obj
    if isinstance(obj,(list,tuple)):return [serial(x,depth+1) for x in obj]
    if isinstance(obj,dict):return {str(k):serial(v,depth+1) for k,v in obj.items()}
    if hasattr(obj,'name') and hasattr(obj,'value'):return str(obj)
    if depth>6:return str(obj)
    result={}
    for name in dir(obj):
        if name.startswith('_'):continue
        try:value=getattr(obj,name)
        except Exception:continue
        if not callable(value):result[name]=serial(value,depth+1)
    return result or str(obj)


def worker(rank,args,commands,replies,go):
    out=Path(args['out']);store=None;control=None;registered=False;record={'rank':rank,'pid':os.getpid(),'role':args['role'],'reads':[]}
    def save():
        temp=out/f'rank{rank}.json.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(out/f'rank{rank}.json')
    try:
        record['phase']='device_init';save()
        import numpy as np
        import torch
        import torch_npu
        torch.set_num_threads(2);torch.npu.set_device(rank)
        from mooncake.store import MooncakeDistributedStore,ReplicateConfig
        record['phase']='store_setup';save()
        store=MooncakeDistributedStore()
        offload=out/'offload'/f'rank{rank}';offload.mkdir(parents=True,exist_ok=False)
        rc=store.setup(args['host'],'P2PHANDSHAKE',512*1024**2 if args['role']=='writer' else 0,
                       1024**3,'ascend','',args['master'],enable_ssd_offload=args['role']=='writer' and args['tier']=='disk',
                       ssd_offload_path=str(offload))
        record['setup_return']=rc;save();assert rc==0,rc
        hostname=store.get_hostname();assert hostname.startswith(args['host']+':')
        size=KEY_BYTES*args['keys']
        owner=torch.empty((size+2*1024**2)//2,dtype=torch.bfloat16,device=f'npu:{rank}')
        byte_owner=owner.view(torch.uint8);offset=(-byte_owner.data_ptr())%(2*1024**2)
        arena=byte_owner[offset:offset+size]
        reference=np.random.default_rng(42+rank).integers(0,256,size=size,dtype=np.uint8).reshape(BUFFERS,args['keys'],BUFFER_BYTES)
        addrs=pointer_lists(arena.data_ptr(),args['keys']);sizes=[[BUFFER_BYTES]*BUFFERS for _ in range(args['keys'])]
        keys=[f"{args['run']}-rank{rank}-key{k}" for k in range(args['keys'])]
        assert store.register_buffer(arena.data_ptr(),size)==0;registered=True
        record.update({'hostname':hostname,'keys':keys,'key_bytes':KEY_BYTES,'arena_bytes':size,
                       'shape':[80,2,args['keys'],128,2,128],'storage_dtype':'bfloat16, uint8 view',
                       'buffer_sizes':sizes,'pointer_offsets':[[p-arena.data_ptr() for p in row] for row in addrs],
                       'phase':'ready'});save()
        if args['clear_via_control'] and args['role']=='writer':
            from worker_control import WorkerControl
            control=WorkerControl(store,args['host']);control_path=control.start(out/'disk')
        else:control_path=None
        replies.put({'rank':rank,'kind':'ready','pid':os.getpid(),'hostname':hostname,'keys':keys,'control_path':control_path})
        def audit(requested):
            descriptors=store.batch_get_replica_desc(requested);result={}
            for key in requested:
                rows=[]
                for d in descriptors.get(key,[]):
                    row={'status':str(d.status),'memory':d.is_memory_replica(),'local_disk':d.is_local_disk_replica()}
                    if row['memory']:row['memory_descriptor']=serial(d.get_memory_descriptor())
                    rows.append(row)
                result[key]=rows
            return result
        while True:
            cmd=commands.get(timeout=900);op=cmd['op'];record['phase']=op;save()
            if op=='stop':break
            if op=='put':
                assert args['role']=='writer'
                arena.copy_(torch.from_numpy(reference.reshape(-1)));torch.npu.synchronize()
                config=ReplicateConfig()
                if hasattr(config,'preferred_segment'):config.preferred_segment=hostname
                record['replicate_config']=serial(config);save()
                values=list(store.batch_put_from_multi_buffers(keys,addrs,sizes,config));torch.npu.synchronize()
                record['put_returns']=values;save();assert values==[0]*len(keys),values
                if control is not None:control.remember(keys,values)
                result={'returns':values}
            elif op=='audit':result=audit(cmd['keys'])
            elif op=='clear':
                result={h:store.batch_replica_clear(cmd['keys'],h) for h in cmd['hostnames']}
            elif op=='clear_own':
                result={'hostname':hostname,'owned_keys':keys,
                        'returns':{h:store.batch_replica_clear(keys,h) for h in cmd['hostnames']}}
                record['last_clear']=result;save()
            elif op=='zero':
                arena.zero_();torch.npu.synchronize();result={'zeroed_bytes':size}
            elif op=='read':
                arena.zero_();torch.npu.synchronize()
                rpc_before=store.get_offload_rpc_read_count()
                replies.put({'rank':rank,'kind':'armed'})
                assert go.wait(timeout=30),'Read cohort was not released'
                begin=time.perf_counter_ns()
                values=list(store.batch_get_into_multi_buffers(keys,addrs,sizes));torch.npu.synchronize()
                end=time.perf_counter_ns();rpc_after=store.get_offload_rpc_read_count()
                actual=arena.cpu().numpy().reshape(BUFFERS,args['keys'],BUFFER_BYTES)
                per_key=[]
                for k,key in enumerate(keys):
                    expected=np.ascontiguousarray(reference[:,k,:]);got=np.ascontiguousarray(actual[:,k,:])
                    per_key.append({'key':key,'returned_bytes':values[k],'expected_bytes':KEY_BYTES,
                        'exact':bool(np.array_equal(got,expected)),
                        'sha256':hashlib.sha256(got.tobytes()).hexdigest(),
                        'expected_sha256':hashlib.sha256(expected.tobytes()).hexdigest()})
                result={'rank':rank,'round':cmd['round'],'begin_ns':begin,'end_ns':end,
                        'elapsed_ms':(end-begin)/1e6,'returns':values,'keys':per_key,
                        'rpc_before':rpc_before,'rpc_after':rpc_after,'rpc_delta':rpc_after-rpc_before}
                record['reads'].append(result);save()
                assert values==[KEY_BYTES]*len(keys),values
                assert all(p['exact'] and p['sha256']==p['expected_sha256'] for p in per_key),'Byte mismatch'
                if args['tier']=='disk':assert rpc_after>rpc_before,'No native SSD read RPC'
            else:raise ValueError(op)
            replies.put({'rank':rank,'kind':'done','result':result})
        record['phase']='closing';save()
    except BaseException:
        record['error']=traceback.format_exc();save();replies.put({'rank':rank,'kind':'error','error':record['error']})
        raise
    finally:
        if store is not None:
            try:
                if control is not None:control.close()
                record['phase']='unregister_begin';save()
                if registered:store.unregister_buffer(arena.data_ptr())
                record['phase']='store_close_begin';save();store.close()
                record['phase']='store_close_end';save()
            except BaseException:
                record['close_error']=traceback.format_exc();save()
                raise
        record['finished_wall_ns']=time.time_ns();save()


def disk_bytes():
    return {p.parent.name:int(p.read_text().split()[2])*512 for p in Path('/sys/block').glob('nvme*n1/stat')}


def emit(value):print(PREFIX+json.dumps(value),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,required=True)
    parser.add_argument('--keys',type=int,required=True);parser.add_argument('--role',choices=['writer','reader'],required=True)
    parser.add_argument('--tier',choices=['memory','disk'],default='disk')
    parser.add_argument('--clear-via-control',action='store_true')
    parser.add_argument('--out',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--host',required=True);parser.add_argument('--master',required=True)
    args=vars(parser.parse_args());assert args['workers'] in [1,4] and args['keys'] in [1,8]
    out=Path(args['out']);out.mkdir(parents=True,exist_ok=True)
    (out/'disk').mkdir(exist_ok=True)
    ctx=mp.get_context('spawn');go=ctx.Event();responses=ctx.Queue();commands=[ctx.Queue() for _ in range(args['workers'])]
    processes=[ctx.Process(target=worker,args=(r,args,commands[r],responses,go)) for r in range(args['workers'])]
    report={'config':args,'key_bytes':KEY_BYTES,'buffers_per_key':BUFFERS,'rounds':[],
            'model_requests':0,'direct_dma_tested':False,'start_wall_ns':time.time_ns()}
    def save(): (out/'layout.json').write_text(json.dumps(report,indent=2))
    def receive(kind,count,timeout=180):
        rows=[];deadline=time.monotonic()+timeout
        while len(rows)<count:
            try:row=responses.get(timeout=min(1,max(.01,deadline-time.monotonic())))
            except queue.Empty:
                dead=[(r,p.pid,p.exitcode) for r,p in enumerate(processes) if p.exitcode is not None]
                assert not dead,('Worker exited',dead)
                if time.monotonic()>deadline:raise TimeoutError('Worker response '+kind)
                continue
            if row['kind']=='error':raise RuntimeError(row['error'])
            assert row['kind']==kind,row;rows.append(row)
        assert len({r['rank'] for r in rows})==count
        return sorted(rows,key=lambda r:r['rank'])
    def dispatch(op,extra=None,rank=None):
        ranks=list(range(args['workers'])) if rank is None else [rank]
        for r in ranks:commands[r].put({'op':op,**(extra or {})})
        return receive('done',len(ranks))
    try:
        for p in processes:p.start()
        ready=receive('ready',args['workers']);report['workers']=ready;save();emit({'kind':'ready','workers':ready})
        allkeys=[k for r in ready for k in r['keys']];hostnames=[r['hostname'] for r in ready]
        for line in sys.stdin:
            cmd=json.loads(line);op=cmd['op']
            if op=='stop':break
            if op=='put':result=dispatch('put')
            elif op=='prepare':
                assert args['role']=='writer' and args['tier']=='disk'
                deadline=time.monotonic()+180
                while True:
                    before=dispatch('audit',{'keys':allkeys},rank=0)[0]['result']
                    if all(v and any(d['local_disk'] and is_complete(d['status']) for d in v) for v in before.values()):break
                    if time.monotonic()>deadline:raise TimeoutError('No COMPLETE LOCAL_DISK for all keys')
                    time.sleep(1)
                time.sleep(6)
                # Master enforces the original publisher's client ID, even when
                # a different client knows the correct segment control name.
                if args['clear_via_control']:
                    from worker_control import request_clear
                    cleared=[request_clear(row['control_path'],allkeys,hostnames) for row in ready]
                    assert set(allkeys)=={k for row in cleared for k in row['owned_keys']}
                else:cleared=dispatch('clear_own',{'hostnames':hostnames})
                after=dispatch('audit',{'keys':allkeys},rank=0)[0]['result']
                report.setdefault('clear_audits',[]).append({'before':before,'clear_results':cleared,'after':after});save()
                assert all(v and all(d['local_disk'] and not d['memory'] and is_complete(d['status']) for d in v) for v in after.values())
                dispatch('zero')
                files=[]
                for p in sorted((out/'offload').rglob('*.bucket')):
                    fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW)
                    try:os.fsync(fd);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                    finally:os.close(fd)
                    files.append({'name':str(p.relative_to(out)),'bytes':p.stat().st_size})
                assert files,'No owned bucket to evict'
                result={'before':before,'lease_wait_s':6,'hostnames':hostnames,'clear_results':cleared,
                        'after':after,'fsync_fadvise_files':files,'global_pagecache_modified':False}
                report.setdefault('pre_read_proofs',[]).append(result);save()
            elif op=='audit':result=dispatch('audit',{'keys':allkeys},rank=0)[0]['result']
            elif op=='read':
                go.clear()
                for q in commands:q.put(cmd)
                receive('armed',args['workers']);before=disk_bytes();begin=time.perf_counter_ns();go.set()
                rows=receive('done',args['workers']);end=time.perf_counter_ns();after=disk_bytes()
                result={'workers':[r['result'] for r in rows],'disk_before':before,'disk_after':after,
                        'counter_begin_ns':begin,'counter_end_ns':end,
                        'disk_read_bytes':sum(after[k]-before[k] for k in before),
                        'counters_scope':'host whole read cohort; overlaps individual rank windows, do not sum'}
                report['rounds'].append(result);save()
            else:raise ValueError(op)
            emit({'kind':'result','op':op,'result':result})
        report['control_complete']=True
    except BaseException:
        report['error']=traceback.format_exc();save();emit({'kind':'error','error':report['error']});raise
    finally:
        for q in commands:q.put({'op':'stop'})
        deadline=time.monotonic()+120
        for p in processes:
            p.join(timeout=max(0,deadline-time.monotonic()))
        for p in processes:
            if p.is_alive():p.terminate();p.join(timeout=20)
            if p.is_alive():p.kill();p.join()
        report['worker_terminals']=[{'rank':r,'pid':p.pid,'exit_code':p.exitcode} for r,p in enumerate(processes)]
        report['finish_wall_ns']=time.time_ns();save()
        emit({'kind':'terminal','workers':report['worker_terminals']})
    assert all(p.exitcode==0 for p in processes),'Nonzero worker exit'


if __name__=='__main__':main()
