"""Verify native Ascend SSD offload with a private master and private directory."""
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import torch
import torch_npu

out = Path(sys.argv[1])
offload = Path(sys.argv[2])
host = sys.argv[3]
report = {'protocol': 'ascend', 'segment_bytes': 64 * 1024**2,
          'payload_bytes': 1024**2, 'ssd_verified': False}
store = None
master = None
owner = None
src = None
dst = None
try:
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    # Native SDK loading must follow NPU runtime initialization.
    from mooncake.store import MooncakeDistributedStore
    offload.mkdir(parents=True, exist_ok=True)
    assert not list(offload.iterdir()), 'Require a new private offload directory'
    settings = {
        'MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES': str(32 * 1024**2),
        'MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES': str(512 * 1024**2),
        'MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE': str(512 * 1024**2),
        'MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES': str(8 * 1024**2),
        'MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT': '1',
        'MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS': '1',
        'MOONCAKE_OFFLOAD_FILE_STORAGE_PATH': str(offload),
        'MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR': 'bucket_storage_backend',
        'MOONCAKE_OFFLOAD_USE_URING': 'false',
    }
    os.environ.update(settings)
    report['offload_settings'] = settings
    for port in (50284, 50285):
        with socket.socket() as sock:
            sock.bind(('0.0.0.0', port))
    command = ['mooncake_master', '--port=50284', '--metrics_port=50285', '--enable_offload=true']
    report['master_command'] = command
    log = (out / 'master.log').open('w')
    master = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(100):
        if master.poll() is not None:
            raise RuntimeError('Master exited: ' + str(master.returncode))
        try:
            with socket.create_connection(('127.0.0.1', 50284), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError('Master readiness timeout')
    store = MooncakeDistributedStore()
    report['setup_doc'] = store.setup.__doc__
    report['setup_return'] = store.setup(host, 'P2PHANDSHAKE', 64 * 1024**2,
        32 * 1024**2, 'ascend', '', '127.0.0.1:50284',
        enable_ssd_offload=True, ssd_offload_path=str(offload))
    assert report['setup_return'] == 0, report['setup_return']
    size = report['payload_bytes']
    reference = torch.arange(256, dtype=torch.uint8).repeat(size // 256)
    owner = torch.empty(size + 2 * 1024**2, dtype=torch.uint8, device='npu:0')
    offset = (-owner.data_ptr()) % (2 * 1024**2)
    src = owner[offset:offset + size]
    src.copy_(reference)
    torch.npu.synchronize()
    assert torch.equal(src.cpu(), reference)
    assert store.register_buffer(src.data_ptr(), size) == 0
    key = 'mac14-ssd-native-' + str(time.time_ns())
    report['put_return'] = store.put_from(key, src.data_ptr(), size)
    assert report['put_return'] == 0, report['put_return']

    def replicas():
        return [{'memory': d.is_memory_replica(), 'local_disk': d.is_local_disk_replica(),
                 'disk': d.is_disk_replica(), 'status': str(d.status)}
                for d in store.get_replica_desc(key)]

    for _ in range(60):
        report['replicas_after_put'] = replicas()
        if any(r['local_disk'] or r['disk'] for r in report['replicas_after_put']):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError('No completed disk replica observed')
    # Replica queries renew the 5000 ms read lease in this runtime.
    # Clearing a leased memory replica is correctly rejected by the master.
    time.sleep(6)
    report['wait_for_lease_expiry_s'] = 6
    report['clear_memory_return'] = store.batch_replica_clear([key], store.get_hostname())
    report['replicas_before_read'] = replicas()
    assert report['replicas_before_read'] and not any(r['memory'] for r in report['replicas_before_read'])
    assert any(r['local_disk'] or r['disk'] for r in report['replicas_before_read'])
    report['files'] = [{'file': str(p.relative_to(offload)), 'bytes': p.stat().st_size,
                        'allocated': p.stat().st_blocks * 512}
                       for p in offload.rglob('*') if p.is_file()]
    report['reads'] = []
    for _ in range(3):
        src.zero_()
        torch.npu.synchronize()
        start = time.perf_counter()
        rc = store.get_into(key, src.data_ptr(), size)
        torch.npu.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        assert rc == size, rc
        assert torch.equal(src.cpu(), reference)
        report['reads'].append({'return_bytes': rc, 'elapsed_ms': elapsed, 'exact': True})
    report['sha256'] = hashlib.sha256(reference.numpy().tobytes()).hexdigest()
    report['ssd_verified'] = True
    assert store.unregister_buffer(src.data_ptr()) == 0
except Exception:
    report['error'] = traceback.format_exc()
    raise
finally:
    if store is not None:
        store.close()
        del store
        gc.collect()
    if master is not None:
        master.terminate()
        try:
            master.wait(timeout=5)
        except subprocess.TimeoutExpired:
            master.kill()
            master.wait()
        report['master_exit_after_cleanup'] = master.returncode
    (out / 'ssd-offload-result.json').write_text(json.dumps(report, indent=2))
