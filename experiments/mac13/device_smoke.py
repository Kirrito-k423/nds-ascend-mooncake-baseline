"""Validate selected A5 cards by exact matmul and one HCCL all-reduce."""
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

rank = int(os.environ['LOCAL_RANK'])
world = int(os.environ['WORLD_SIZE'])
torch.npu.set_device(rank)
torch.manual_seed(42 + rank)
a = torch.randint(-2, 3, (64, 64), dtype=torch.int32)
b = torch.randint(-2, 3, (64, 64), dtype=torch.int32)
expected = a.double() @ b.double()
actual = (a.to(f'npu:{rank}', dtype=torch.float16)
          @ b.to(f'npu:{rank}', dtype=torch.float16)).cpu().double()
assert torch.equal(expected, actual), 'NPU matrix multiplication mismatch'
dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=90))
x = torch.full((4096,), float(rank), device=f'npu:{rank}')
start = time.perf_counter()
dist.all_reduce(x)
torch.npu.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1000
assert torch.equal(x.cpu(), torch.full((4096,), float(world * (world - 1) // 2)))
free, total = torch.npu.mem_get_info()
record = {'rank': rank, 'world_size': world, 'integer_matmul_exact': True,
          'all_reduce_correct': True, 'all_reduce_ms_with_first_call': elapsed_ms,
          'device_name': torch.npu.get_device_name(rank),
          'total_hbm_bytes': total, 'free_hbm_bytes': free,
          'smoke_peak_allocated_bytes': torch.npu.max_memory_allocated(),
          'smoke_peak_reserved_bytes': torch.npu.max_memory_reserved()}
Path(f'/evidence/smoke-rank{rank}.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
dist.destroy_process_group()
