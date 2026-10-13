"""Audit real KV Pool SDK buffers. Enabled only for the isolated smoke run.

Observe the runtime's registered NPU storage; never substitute a cache layout.
Hash every transferred byte, poison the first 64 destination bytes before get,
then compare restored buffers with the corresponding successful put.
This intentionally adds synchronization/copies and is not a performance test.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get('MAC17_EVIDENCE', '/evidence'))
EVENTS = ROOT / 'pool-bytes.jsonl'
REGIONS = []


def event(value):
    value['pid'] = os.getpid()
    fd = os.open(EVENTS, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(value) + '\n').encode())
    finally:
        os.close(fd)


def raw(ptr, size):
    for base, length, tensor in REGIONS:
        if base <= ptr and ptr + size <= base + length:
            view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            view.set_(tensor.untyped_storage(), 0, (length,), (1,))
            return view[ptr-base:ptr-base+size]
    raise RuntimeError(f'Transfer range not found in actual KV storage: size={size}')


def hashes(addrs, sizes):
    torch.npu.synchronize()
    return [hashlib.sha256(raw(int(p), int(n)).cpu().numpy().tobytes()).hexdigest()
            for p, n in zip(addrs, sizes)]


def references():
    result = {}
    if EVENTS.exists():
        for line in EVENTS.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A different rank may be appending its single line.
            if row.get('operation') == 'put' and row['sdk_result'] >= 0:
                result[row['key']] = row
    return result


class StoreProxy:
    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        before = [hashes(p, n) for p, n in zip(addrs, sizes)]
        result = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
        for key, ns, digest, rc in zip(keys, sizes, before, result):
            event({'operation': 'put', 'key': key, 'sizes': list(ns),
                   'sha256': digest, 'sdk_result': int(rc), 'full_bytes_hashed': True})
        return result

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        expected = references()
        poisoned = []
        for key, ps, ns in zip(keys, addrs, sizes):
            matched = key in expected and list(ns) == expected[key]['sizes']
            if matched:
                for p, n in zip(ps, ns):
                    raw(int(p), int(n))[:min(64, int(n))].fill_(0xA5)
            poisoned.append(matched)
        torch.npu.synchronize()
        result = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
        torch.npu.synchronize()
        for key, ps, ns, poison, rc in zip(keys, addrs, sizes, poisoned, result):
            digest = hashes(ps, ns) if int(rc) >= 0 else None
            event({'operation': 'get', 'key': key, 'sizes': list(ns), 'sha256': digest,
                   'sdk_result': int(rc), 'destination_poisoned': poison,
                   'full_bytes_hashed': True,
                   'matches_put': digest == expected[key]['sha256'] if poison else None})
        return result


def install():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend
    register = KVPoolWorker.register_kv_caches
    initialize = MooncakeBackend.ensure_initialized

    def observed_register(self, caches):
        for value in caches.values():
            values = value if isinstance(value, (list, tuple)) else [value]
            for tensor in values:
                storage = tensor.untyped_storage()
                region = (storage.data_ptr(), storage.nbytes(), tensor)
                if not any(r[0] == region[0] for r in REGIONS):
                    REGIONS.append(region)
        result = register(self, caches)
        event({'operation': 'register', 'num_groups': self.num_kv_cache_groups,
               'group_families': self.group_kv_cache_families,
               'group_block_len': self.group_block_len,
               'group_block_stride': self.group_block_stride,
               'unique_region_bytes': sum(r[1] for r in REGIONS)})
        return result

    def observed_initialize(self):
        result = initialize(self)
        if self.store is not None and not isinstance(self.store, StoreProxy):
            self.store = StoreProxy(self.store)
        return result

    KVPoolWorker.register_kv_caches = observed_register
    MooncakeBackend.ensure_initialized = observed_initialize
