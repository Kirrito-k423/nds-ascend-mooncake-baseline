"""Positive minimal DSA metadata probe; no model or cache capacity claim."""
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend import ops
from vllm_ascend.utils import bootstrap_custom_op_env

torch.npu.set_device(0)
ops.register_dummy_fusion_op()
bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C
op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata
cu=torch.tensor([0,2048],dtype=torch.int32,device='npu:0')
used=torch.tensor([2048],dtype=torch.int32,device='npu:0')
records=[]
for ratio in [1,4,128]:
    record={'num_heads_q':64,'num_heads_kv':1,'head_dim':512,'cmp_ratio':ratio,
            'query_tokens':2048,'stage':'metadata_only'}
    out=op(num_heads_q=64,num_heads_kv=1,head_dim=512,kv_quant_mode=1,
           cu_seqlens_q=cu,cu_seqlens_ori_kv=cu,cu_seqlens_cmp_kv=None,
           seqused_q=used,seqused_kv=used,max_seqlen_q=2048,max_seqlen_kv=2048,
           batch_size=1,cmp_topk=512 if ratio==4 else 0,cmp_ratio=ratio,
           ori_mask_mode=4,cmp_mask_mode=3,ori_win_left=127,ori_win_right=0,
           layout_q='TND',layout_kv='PA_ND',has_ori_kv=True,has_cmp_kv=ratio>1)
    torch.npu.synchronize()
    record.update(passed=True,shape=list(out.shape),dtype=str(out.dtype),
                  metadata_values=out.cpu().tolist())
    records.append(record)
Path('/evidence/metadata-probe.json').write_text(json.dumps(records,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k!='metadata_values'} for r in records]),flush=True)
