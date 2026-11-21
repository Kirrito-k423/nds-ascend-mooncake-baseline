"""Record installed source and validate config/primitive quantization support."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

root=Path('/evidence');root.mkdir(parents=True,exist_ok=True)
versions={}
for package in ['torch','torch-npu','vllm','vllm-ascend','transformers','tokenizers','mooncake-transfer-engine-npu']:
    try:versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:versions[package]=None
(root/'versions.json').write_text(json.dumps(versions,indent=2))
from vllm.model_executor.models import ModelRegistry
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained('/model',local_files_only=True,trust_remote_code=False)
    direct_config_recognized=True
except ValueError:
    # vLLM ships its own HF config registry for models absent in Transformers.
    direct_config_recognized=False
args=EngineArgs(model='/model',skip_tokenizer_init=True,load_format='dummy',dtype='bfloat16',
    quantization='fp8',tensor_parallel_size=1,data_parallel_size=8,enable_expert_parallel=True,
    enforce_eager=True,max_model_len=8224,max_num_seqs=1,max_num_batched_tokens=2048,
    enable_prefix_caching=False,disable_hybrid_kv_cache_manager=False,
    speculative_config={'method':'mtp','num_speculative_tokens':1,'enforce_eager':True})
engine_config=args.create_engine_config()
config=engine_config.model_config.hf_config
record={'config_recognized':True,'model_type':config.model_type,
    'direct_transformers_config_recognized':direct_config_recognized,
    'architectures':config.architectures,'num_hidden_layers':config.num_hidden_layers,
    'num_nextn_predict_layers':config.num_nextn_predict_layers,
    'target_registered':all(a in ModelRegistry.get_supported_archs() for a in config.architectures),
    'effective_quantization':engine_config.model_config.quantization,
    'quantization_class':type(engine_config.quant_config).__name__,
    'quantization_config':config.quantization_config,
    'compress_ratios':config.compress_ratios}
sources={}
files={'vllm_ascend':['models/deepseek_v4.py','models/deepseek_v4_mtp.py','worker/worker.py',
    'worker/model_runner_v1.py','quantization/fp8_config.py','quantization/methods/fp8.py',
    'quantization/methods/w4a8_mxfp4.py','quantization/methods/w8a8_mxfp8.py'],
    'vllm':['model_executor/model_loader/dummy_loader.py','model_executor/model_loader/weight_utils.py']}
for package,paths in files.items():
    base=Path(importlib.util.find_spec(package).origin).parent
    for relative in paths:
        path=base/relative
        if path.exists():
            data=path.read_bytes();target=root/'installed-source'/package/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            sources[package+'/'+relative]=hashlib.sha256(data).hexdigest()
record['source_sha256']=sources
(root/'version-probe.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)

import torch,torch_npu
torch.npu.set_device(0)
scale=torch.empty(4,128,dtype=torch.float8_e8m0fnu,device='npu:0')
scale.view(torch.uint8).fill_(127)
scale_bits=scale.view(torch.uint8).cpu()
# The runtime uses UE8M0 as raw exponent bits. CANN 9.1 rejects an unused
# UE8M0->float D2D cast, so decode the same bits on CPU for this assertion.
assert torch.equal(scale_bits.view(torch.float8_e8m0fnu).float(),torch.ones(4,128)), 'UE8M0 neutral scale wrong'
packed=torch.full((1,32,64),34,dtype=torch.uint8,device='npu:0')
converted=torch_npu.npu_format_cast(packed,29,customize_dtype=torch.float8_e4m3fn,
                                  input_dtype=torch_npu.float4_e2m1fn_x2)
torch.npu.synchronize()
# Explicitly return to ND before copying a tensor in internal NZ format.
converted_nd=torch_npu.npu_format_cast(converted,2)
primitive={'e8m0_scale_1_correct':True,'packed_fp4_shape':list(packed.shape),
    'packed_fp4_dtype':str(packed.dtype),'converted_shape':list(converted.shape),
    'converted_dtype':str(converted.dtype),
    'finite_converted':bool(torch.isfinite(converted_nd.view(torch.uint8).cpu().view(torch.float8_e4m3fn).float()).all().item())}
assert primitive['finite_converted'], 'nonfinite converted fake FP4'
(root/'quant-primitive.json').write_text(json.dumps(primitive,indent=2));print(json.dumps(primitive),flush=True)
