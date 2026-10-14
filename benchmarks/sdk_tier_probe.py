"""Small single-NPU MEMORY/LOCAL_DISK readback; not model TTFT or direct DMA."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch_npu

torch.set_num_threads(2)
torch.npu.set_device(0)
# Keep native import after device selection.
from mooncake.store import MooncakeDistributedStore

mode=sys.argv[1]
assert mode in ['memory','disk']
out=Path('/evidence')
store=MooncakeDistributedStore()
sizes=json.loads(os.environ.get('NDS_PROBE_SIZES_JSON','[1048576,3145728]'))
assert sizes and all(isinstance(x,int) and 0<x for x in sizes)
if mode=='disk' and os.environ.get('NDS_BOUNDARY_PROBE')!='1':
    assert all(x<=3*1024**2 for x in sizes), 'Unvalidated SSD object size; explicit boundary probe required'
report={'mode':mode,'sizes':sizes,'rounds':3,'records':[],
        'reference':'numpy PCG64 seed 42+size, uint8 positional random values',
        'model_requests':0,'direct_dma_tested':False,'start_wall_ns':time.time_ns()}

def save():
    (out/('native-'+mode+'.json')).write_text(json.dumps(report,indent=2))

def serial(obj,depth=0):
    if obj is None or isinstance(obj,(str,int,float,bool)):return obj
    if isinstance(obj,(list,tuple)):return [serial(x,depth+1) for x in obj]
    if isinstance(obj,dict):return {str(k):serial(v,depth+1) for k,v in obj.items()}
    if depth>6:return str(obj)
    values={}
    for name in dir(obj):
        if name.startswith('_'):continue
        try:value=getattr(obj,name)
        except Exception:continue
        if not callable(value):values[name]=serial(value,depth+1)
    return values or str(obj)

def audit(key):
    rows=[]
    for d in store.batch_get_replica_desc([key]).get(key,[]):
        r={'status':str(d.status),'memory':d.is_memory_replica(),
           'local_disk':d.is_local_disk_replica()}
        if r['memory']:r['memory_descriptor']=serial(d.get_memory_descriptor())
        if r['local_disk']:
            r['descriptor_methods']=[n for n in dir(d) if 'descriptor' in n]
            if hasattr(d,'get_local_disk_descriptor'):
                r['disk_descriptor']=serial(d.get_local_disk_descriptor())
            else:
                r['local_disk_descriptor_not_exposed']=True
        rows.append(r)
    return rows

def segments(value):
    names=set()
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ['segment_name','transport_endpoint'] and isinstance(v,str):names.add(v)
            else:names.update(segments(v))
    elif isinstance(value,list):
        for v in value:names.update(segments(v))
    return names

def complete(row):return row['status'].rsplit('.',1)[-1]=='COMPLETE'

def only_disk(key):
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        before=audit(key)
        if any(r['local_disk'] and complete(r) for r in before):break
        time.sleep(1)
    else:
        report['disk_wait_timeout']={'key':key,'last_replicas':before,'wait_s':120}
        save()
        raise TimeoutError('No COMPLETE LOCAL_DISK in 120s')
    descriptor_endpoints=segments([r.get('memory_descriptor') for r in before if r['memory']])
    # This private master has exactly one contributing client. The SDK's
    # control hostname is the segment name; a TE endpoint uses another port.
    names={store.get_hostname()} if any(r['memory'] for r in before) else set()
    report['last_disk_clear']={'replicas':before,'segments':sorted(names),
                               'client_hostname':store.get_hostname(),
                               'descriptor_transport_endpoints':sorted(descriptor_endpoints),
                               'name_source':'single contributing store.get_hostname'}
    save()
    assert names or not any(r['memory'] for r in before)
    time.sleep(6)
    cleared={s:store.batch_replica_clear([key],s) for s in sorted(names)}
    after=audit(key)
    assert after and all(r['local_disk'] and not r['memory'] and complete(r) for r in after)
    evicted=[]
    if os.environ.get('NDS_OWN_FILE_COLD')=='1':
        for p in Path('/evidence/offload-disk').glob('*.bucket'):
            fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
            evicted.append(p.name)
    return {'before':before,'after':after,'lease_wait_s':6,'cleared':cleared,
            'own_files_fsync_fadvise':evicted,'global_page_cache_modified':False}

def disk_bytes():
    return {p.parent.name:int(p.read_text().split()[2])*512
            for p in Path('/sys/block').glob('nvme*n1/stat')}

def aligned(size):
    owner=torch.empty(size+2*1024**2,dtype=torch.uint8,device='npu:0')
    offset=(-owner.data_ptr())%(2*1024**2)
    return owner,owner[offset:offset+size]

try:
    offload=Path('/evidence/offload-'+mode)
    offload.mkdir()
    rc=store.setup(os.environ['NDS_LOCAL_HOST'],'P2PHANDSHAKE',256*1024**2,32*1024**2,
                   'ascend','',os.environ['NDS_MASTER'],enable_ssd_offload=mode=='disk',
                   ssd_offload_path='/evidence/offload-'+mode)
    report['setup_return']=rc;save();assert rc==0,rc
    time.sleep(6)
    removed=store.remove_all();report['remove_all_return']=removed;save()
    assert removed>=0
    for size in report['sizes']:
        reference=np.random.default_rng(42+size).integers(0,256,size=size,dtype=np.uint8)
        expected_sha=hashlib.sha256(reference.tobytes()).hexdigest()
        owner,buf=aligned(size)
        buf.copy_(torch.from_numpy(reference));torch.npu.synchronize()
        assert torch.equal(buf.cpu(),torch.from_numpy(reference))
        assert store.register_buffer(buf.data_ptr(),size)==0
        key='mac16-sdk-'+mode+'-'+str(size)
        try:
            assert store.put_from(key,buf.data_ptr(),size)==0
            for rn in range(3):
                proof=only_disk(key) if mode=='disk' else {'after':audit(key)}
                if mode=='memory':
                    assert proof['after'] and all(r['memory'] and not r['local_disk'] and complete(r)
                                                for r in proof['after'])
                buf.zero_();torch.npu.synchronize()
                before=disk_bytes();rpc_before=store.get_offload_rpc_read_count()
                begin=time.perf_counter_ns()
                returned=store.get_into(key,buf.data_ptr(),size);torch.npu.synchronize()
                elapsed=(time.perf_counter_ns()-begin)/1e6
                rpc_after=store.get_offload_rpc_read_count();after=disk_bytes()
                actual=buf.cpu().numpy()
                record={'size_bytes':size,'round':rn,'replicas':proof,'returned_bytes':returned,
                        'elapsed_ms':elapsed,'sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
                        'expected_sha256':expected_sha,'exact':bool(np.array_equal(actual,reference)),
                        'offload_rpc_delta':rpc_after-rpc_before,'disk_before':before,'disk_after':after,
                        'disk_read_bytes':sum(after[k]-before[k] for k in before)}
                record['physical_disk_read_witness']=mode=='disk' and record['disk_read_bytes']>=size
                report['records'].append(record);save()
                assert returned==size and record['exact'] and record['sha256']==expected_sha
                if mode=='disk':assert record['offload_rpc_delta']>0,'No SSD offload read RPC'
        finally:
            assert store.unregister_buffer(buf.data_ptr())==0
    report['functional_pass']=True
except Exception:
    import traceback
    report['error']=traceback.format_exc();raise
finally:
    store.close()
    report['finish_wall_ns']=time.time_ns();save()
