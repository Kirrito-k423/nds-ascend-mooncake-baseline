"""Lease-aware metadata and exact readback service for an isolated master."""
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from worker_control import request_clear

import numpy as np
import torch
import torch_npu

torch.set_num_threads(2)
torch.npu.set_device(0)
# Preserve the validated native import ordering.
from mooncake.store import MooncakeDistributedStore

STORE = MooncakeDistributedStore()
assert STORE.setup(os.environ['NDS_LOCAL_HOST'], 'P2PHANDSHAKE', 0, 32*1024**2,
                   'ascend', '', os.environ['NDS_MASTER']) == 0
OUT = Path('/evidence')
PROBE_KEYS=set()


def serial(obj, depth=0):
    if obj is None or isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, (list, tuple)): return [serial(x, depth+1) for x in obj]
    if isinstance(obj, dict): return {str(k): serial(v, depth+1) for k, v in obj.items()}
    if hasattr(obj,'name') and hasattr(obj,'value'):return str(obj)
    if depth > 8: return str(obj)
    result = {}
    for name in dir(obj):
        if name.startswith('_'): continue
        try: value = getattr(obj, name)
        except Exception: continue
        if not callable(value): result[name] = serial(value, depth+1)
    return result or str(obj)


def audit(keys):
    descriptors = STORE.batch_get_replica_desc(keys)
    result = {}
    for key in keys:
        rows = []
        for d in descriptors.get(key, []):
            row = {'status': str(d.status), 'memory': d.is_memory_replica(),
                   'local_disk': d.is_local_disk_replica(), 'disk': d.is_disk_replica()}
            for name in ['get_memory_descriptor', 'get_local_disk_descriptor', 'get_disk_descriptor']:
                flag = {'get_memory_descriptor': 'memory', 'get_local_disk_descriptor': 'local_disk',
                        'get_disk_descriptor': 'disk'}[name]
                if row[flag] and hasattr(d, name): row[flag+'_descriptor'] = serial(getattr(d, name)())
            rows.append(row)
        result[key] = rows
    return result


def segment_names(value):
    names = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k == 'segment_name' and isinstance(v, str): names.add(v)
            else: names.update(segment_names(v))
    elif isinstance(value, list):
        for v in value: names.update(segment_names(v))
    return names


def complete(status):
    return status.rsplit('.',1)[-1] == 'COMPLETE'


def disk_only(keys,known_segments=None,worker_controls=None):
    before = audit(keys)
    assert keys and all(rows and any(r['local_disk'] and complete(r['status']) for r in rows)
                        for rows in before.values()), 'Missing COMPLETE LOCAL_DISK'
    segments = segment_names([r['memory_descriptor'] for rows in before.values()
                              for r in rows if r['memory']])
    if known_segments is not None:
        assert known_segments and all(isinstance(s,str) and s.startswith(os.environ['NDS_LOCAL_HOST']+':')
                                      for s in known_segments),'Require task-owned source control hostnames'
        segments.update(known_segments)
    if any(r['memory'] for rows in before.values() for r in rows):
        assert segments, 'Memory descriptor has no segment_name; refuse unknown clearing'
    # No replica query may occur between this wait and clearing.
    time.sleep(6)
    cleared=[];covered=set()
    if worker_controls is not None:
        assert set(worker_controls)==set(known_segments),'Worker/client map mismatch'
        for path in worker_controls.values():
            result=request_clear(path,keys,sorted(segments));cleared.append(result);covered.update(result['owned_keys'])
    own_probe=sorted(set(keys)&PROBE_KEYS)
    if own_probe:
        cleared.append({'agent_owned_keys':own_probe,'clear_results':{s:STORE.batch_replica_clear(own_probe,s) for s in sorted(segments)}})
        covered.update(own_probe)
    assert covered==set(keys),'Refuse keys without their original publisher client'
    after = audit(keys)
    (OUT/'last-disk-clear.json').write_text(json.dumps({'before':before,'clear_results':cleared,'after':after},indent=2))
    assert all(rows and all(not r['memory'] and r['local_disk'] and complete(r['status']) for r in rows)
               for rows in after.values()), 'A COMPLETE LOCAL_DISK-only replica is required'
    evicted=[]
    if os.environ.get('NDS_OWN_FILE_COLD')=='1':
        # /offload is the empty task-owned bind mount created by Node.start.
        # Never clear the host-wide page cache or touch another task's files.
        # Model workers write to independent subdirectories under this mount.
        for p in Path('/offload').rglob('*.bucket'):
            fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
            evicted.append(str(p.relative_to('/offload')))
        assert evicted,'No owned bucket found for cold-file preparation'
    return {'before': before, 'lease_wait_s': 6, 'segments': sorted(segments),
            'clear_results': cleared, 'after': after,
            'own_files_fsync_fadvise':evicted,'global_page_cache_modified':False}


def aligned(size):
    owner = torch.empty(size+2*1024**2, dtype=torch.uint8, device='npu:0')
    offset = (-owner.data_ptr()) % (2*1024**2)
    return owner, owner[offset:offset+size]


def reference(size):
    # Position-sensitive data also detects whole 256-byte block reordering.
    values=np.random.default_rng(42+size).integers(0,256,size=size,dtype=np.uint8)
    return torch.from_numpy(values)


def probe_put(size):
    owner, data = aligned(size)
    host = reference(size)
    data.copy_(host); torch.npu.synchronize()
    assert torch.equal(data.cpu(), host)
    key = 'mac16-native-'+str(time.time_ns())
    assert STORE.register_buffer(data.data_ptr(), size) == 0
    try: rc = STORE.put_from(key, data.data_ptr(), size)
    finally: assert STORE.unregister_buffer(data.data_ptr()) == 0
    assert rc == 0, rc
    PROBE_KEYS.add(key)
    return {'key': key, 'size': size, 'sha256': hashlib.sha256(host.numpy().tobytes()).hexdigest()}


def probe_read(key, size):
    owner, data = aligned(size)
    data.zero_(); torch.npu.synchronize()
    assert STORE.register_buffer(data.data_ptr(), size) == 0
    try:
        start = time.perf_counter_ns()
        rc = STORE.get_into(key, data.data_ptr(), size)
        torch.npu.synchronize()
        elapsed = (time.perf_counter_ns()-start)/1e6
        assert rc == size, (rc, size)
        actual = data.cpu(); expected = reference(size)
        exact = torch.equal(actual, expected)
        assert exact, 'KV readback mismatch'
        return {'key': key, 'bytes': rc, 'elapsed_ms': elapsed, 'exact': exact,
                'sha256': hashlib.sha256(actual.numpy().tobytes()).hexdigest()}
    finally: assert STORE.unregister_buffer(data.data_ptr()) == 0


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            path = self.path
            if path == '/audit': result = audit(body['keys'])
            elif path == '/disk-only': result = disk_only(body['keys'],body.get('segments'),body.get('worker_controls'))
            elif path == '/remove-all':
                # Isolated task-owned master only. Never use against a shared pool.
                time.sleep(6)
                result = {'return': STORE.remove_all(), 'lease_wait_s':6}
            elif path == '/probe-put': result = probe_put(body['size'])
            elif path == '/probe-read': result = probe_read(body['key'], body['size'])
            else: raise ValueError('Unknown action')
            code = 200
        except Exception:
            code = 500; result = {'error': traceback.format_exc()}
        payload = json.dumps(result).encode()
        self.send_response(code); self.end_headers(); self.wfile.write(payload)

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')

    def log_message(self, *args): pass


try:
    HTTPServer(('127.0.0.1', int(os.environ['NDS_AGENT_PORT'])), Handler).serve_forever()
finally:
    STORE.close(); del STORE; gc.collect()
