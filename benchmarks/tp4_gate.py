import json, os, pathlib, time
import torch
import torch_npu
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(rank)
torch.manual_seed(42 + rank)
a = torch.randint(-2, 3, (64, 64), dtype=torch.int32)
b = torch.randint(-2, 3, (64, 64), dtype=torch.int32)
expected = a.double() @ b.double()
actual = (a.to(f'npu:{rank}', dtype=torch.float16) @ b.to(f'npu:{rank}', dtype=torch.float16)).cpu().double()
assert torch.equal(expected, actual), 'NPU matrix multiplication data mismatch'
dist.init_process_group('hccl')
x = torch.full((4096,), float(rank), device=f'npu:{rank}')
dist.all_reduce(x)
torch.npu.synchronize()
assert torch.equal(x.cpu(), torch.full((4096,), 6.0))
for _ in range(5):
    dist.all_reduce(x)
torch.npu.synchronize()
start = time.perf_counter()
for _ in range(20):
    dist.all_reduce(x)
torch.npu.synchronize()
pathlib.Path(f'/evidence/tp4-rank{rank}.json').write_text(json.dumps({
    'pid':os.getpid(), 'completed_wall_ns':time.time_ns(), 'rank':rank, 'world_size':dist.get_world_size(), 'all_reduce_correct':True,
    'integer_matmul_exact':True,
    'mean_ms':(time.perf_counter()-start)*1000/20,
    'device_name':torch.npu.get_device_name(rank)}, indent=2))
dist.destroy_process_group()
