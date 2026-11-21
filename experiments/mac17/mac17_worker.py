"""Observe original allocation and initialize deterministic, valid fake tensors.

Only dummy numeric values change. Model construction, tensor shape/dtype,
quantization processing and cache allocation remain runtime-owned.
"""
import dataclasses
import json
import os
import time
import traceback
from pathlib import Path

import torch
from vllm_ascend.worker.worker import NPUWorker


def serial(value):
    if dataclasses.is_dataclass(value):
        return {f.name: serial(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [serial(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(value)


def memory():
    torch.npu.synchronize()
    free,total=torch.npu.mem_get_info()
    return {'device_free_bytes':free,'device_total_bytes':total,
            'allocated_bytes':torch.npu.memory_allocated(),
            'reserved_bytes':torch.npu.memory_reserved(),
            'peak_allocated_bytes':torch.npu.max_memory_allocated(),
            'peak_reserved_bytes':torch.npu.max_memory_reserved()}


def evidence_rank(parallel_config):
    """DP workers can each expose their physical NPU as local device 0."""
    from vllm.distributed import get_tensor_model_parallel_rank
    return int(parallel_config.data_parallel_rank)*int(parallel_config.tensor_parallel_size)+get_tensor_model_parallel_rank()


def tensors(model):
    result=[];seen={};logical=0;storage=0;cpu=0
    cpu_parameters=sum(t.numel()*t.element_size() for t in model.parameters() if t.device.type=='cpu')
    for name,t in list(model.named_parameters())+list(model.named_buffers()):
        s=t.untyped_storage();key=(str(t.device),s.data_ptr())
        duplicate=seen.get(key)
        if duplicate is None:
            seen[key]=name;storage+=s.nbytes()
        b=t.numel()*t.element_size();logical+=b
        if t.device.type=='cpu':cpu+=b
        result.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),
            'device':str(t.device),'numel':t.numel(),'logical_bytes':b,
            'reported_storage_bytes':s.nbytes(),'storage_alias_of':duplicate})
    return {'tensors':result,'logical_bytes_with_views':logical,
            'unique_reported_storage_bytes':storage,'cpu_tensor_logical_bytes':cpu,
            'cpu_parameter_logical_bytes':cpu_parameters}


@torch.no_grad()
def initialize_fake(model, config):
    """No integer garbage, no negative/NaN exponent scales, no shape changes."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.config import get_current_vllm_config
    rank=evidence_rank(get_current_vllm_config().parallel_config)
    root=Path(os.environ.get('MAC17_EVIDENCE','/evidence'))
    root.mkdir(parents=True,exist_ok=True)
    kind=type(model).__name__
    (root/f'created-rank{rank}-{kind}.json').write_text(json.dumps(tensors(model),indent=2))
    initialized=set()
    for name,param in model.named_parameters():
        key=(str(param.device),param.untyped_storage().data_ptr())
        if key in initialized:continue
        initialized.add(key)
        dtype=str(param.dtype)
        if name.endswith('tid2eid'):
            # Every token maps to distinct, legal experts; keep original int32 shape.
            row=torch.arange(param.shape[-1],device=param.device,dtype=param.dtype)
            param.copy_(row.expand_as(param))
        elif dtype=='torch.float8_e8m0fnu':
            # UE8M0 stores exponent 127 for 1.0; 255 would encode NaN.
            param.view(torch.uint8).fill_(127)
        elif 'float8' in dtype:
            # Finite, small positive E4M3 weight bits; original FP8 layout.
            param.view(torch.uint8).fill_(1)
        elif not torch.is_floating_point(param):
            # Native expert weights are packed 2xFP4 in one I8/U8 element.
            param.fill_(34 if 'weight' in name else 0)
        elif 'weight_scale' in name or name.endswith('_scale'):
            param.fill_(1.0)
        elif param.ndim==1 and 'norm' in name and name.endswith('weight'):
            param.fill_(1.0)
        else:
            param.fill_(1e-4)
    (root/f'fake-init-rank{rank}-{kind}.json').write_text(json.dumps({
        'model_class':kind,'initialized_unique_parameter_storages':len(initialized),
        'numeric_values_only_changed':True,'scale_e8m0_bits':127,
        'packed_fp4_bits':34,'hash_expert_ids':'distinct range(num_experts_per_tok)',
        'random_business_weights':False},indent=2))


class AuditWorker(NPUWorker):
    def record(self,stage,**extra):
        device_id=evidence_rank(self.parallel_config)
        path=Path(os.environ.get('MAC17_EVIDENCE','/evidence'))/f'audit-rank{device_id}.jsonl'
        value={'stage':stage,'rank':device_id,'worker_rank':self.rank,
               'local_device_ordinal':int(torch.npu.current_device()),
               'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
               'dp_rank':getattr(self.parallel_config,'data_parallel_rank',None),'time_ns':time.time_ns(),
               'monotonic_s':time.monotonic(),'memory':memory(),**extra}
        with path.open('a') as f:f.write(json.dumps(serial(value))+'\n')

    def load_model(self):
        if os.environ.get('MAC17_POOL_BYTE_AUDIT') == '1':
            from pool_audit import install
            install()
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
        def fake_load(loader,model,model_config):initialize_fake(model,model_config)
        DummyModelLoader.load_weights=fake_load
        self.record('before_load')
        try:
            super().load_model()
            model=self.model_runner.model
            info=tensors(model)
            decoder_layers=len(getattr(getattr(model,'model',None),'layers',[]))
            device_id=evidence_rank(self.parallel_config)
            Path(f'/evidence/loaded-rank{device_id}-main.json').write_text(json.dumps(info,indent=2))
            draft=getattr(getattr(self.model_runner,'drafter',None),'model',None)
            draft_info=tensors(draft) if isinstance(draft,torch.nn.Module) else None
            if draft_info is not None:
                Path(f'/evidence/loaded-rank{device_id}-draft.json').write_text(json.dumps(draft_info,indent=2))
            assert info['cpu_parameter_logical_bytes']==0,'Main weights retained on CPU'
            if draft_info is not None:
                assert draft_info['cpu_parameter_logical_bytes']==0,'Draft weights retained on CPU'
            self.record('after_load',model_memory_usage=getattr(self.model_runner,'model_memory_usage',None),
                        main_decoder_layer_count=decoder_layers,
                        main_unique_reported_storage_bytes=info['unique_reported_storage_bytes'],
                        draft_unique_reported_storage_bytes=draft_info['unique_reported_storage_bytes'] if draft_info else None)
        except Exception as exc:
            self.record('load_failed',exception_type=type(exc).__name__,exception_message=str(exc))
            raise

    def determine_available_memory(self):
        self.record('before_profile')
        try:
            available=super().determine_available_memory()
            self.record('after_profile',available_kv_cache_memory_bytes=available,
                        peak_activation_memory=getattr(self,'peak_activation_memory',None),
                        non_torch_memory=getattr(self,'non_torch_memory',None),
                        requested_memory=getattr(self,'requested_memory',None))
            return available
        except Exception as exc:
            self.record('profile_failed',exception_type=type(exc).__name__,exception_message=str(exc))
            raise

    def initialize_from_config(self,kv_cache_config):
        self.record('before_kv',kv_config=kv_cache_config)
        try:
            super().initialize_from_config(kv_cache_config)
            layers={};seen={};total=0
            def visit(name,value):
                nonlocal total
                if isinstance(value,torch.Tensor):
                    s=value.untyped_storage();key=(str(value.device),s.data_ptr())
                    alias=seen.get(key)
                    if alias is None:seen[key]=name;total+=s.nbytes()
                    layers[name]={'shape':list(value.shape),'dtype':str(value.dtype),
                        'logical_bytes':value.numel()*value.element_size(),
                        'reported_storage_bytes':s.nbytes(),'alias_of':alias}
                elif isinstance(value,(list,tuple)):
                    for i,v in enumerate(value):visit(f'{name}/{i}',v)
            for name,layer in self.model_runner.compilation_config.static_forward_context.items():
                visit(name,getattr(layer,'kv_cache',None))
            self.record('after_kv',layers=layers,unique_kv_reported_storage_bytes=total,
                        kv_config=kv_cache_config)
        except Exception as exc:
            self.record('kv_failed',exception_type=type(exc).__name__,exception_message=str(exc))
            raise
