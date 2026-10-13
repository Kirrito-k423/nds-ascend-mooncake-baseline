"""Minimal full-target architecture startup; W8A8 config is unmodified.

This is not a full W8A8 checkpoint emulation or a weight-capacity test. It detects
the first actual runtime failure before a W8A8 experiment can be constructed.
"""
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

packages = ['torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers', 'tokenizers']
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
Path('/evidence/versions.json').write_text(json.dumps(versions, indent=2))
print(json.dumps(versions), flush=True)
Path('/evidence/python-runtime.json').write_text(json.dumps({
    'python': platform.python_version(),
    'vllm_effective_version_env': os.environ.get('VLLM_VERSION'),
    'cann_latest_target': str(Path('/usr/local/Ascend/ascend-toolkit/latest').resolve()),
}, indent=2))
settings = dict(model='/model', load_format='dummy', skip_tokenizer_init=True,
                dtype='bfloat16', tensor_parallel_size=8,
                data_parallel_size=1, enable_expert_parallel=True,
                max_model_len=8192, max_num_seqs=1,
                max_num_batched_tokens=8192, gpu_memory_utilization=0.9,
                kv_cache_dtype='auto', quantization='ascend',
                enforce_eager=True, cpu_offload_gb=0,
                enable_prefix_caching=False)
record = {'probe_kind': 'architecture_startup_with_original_w8a8_config',
          'settings': settings, 'w8a8_metadata_available_locally': True,
          'remote_metadata_scope': 'unmodified config only; no tensors allocated',
          'weight_capacity_measured': False, 'requests_attempted': 0}
start = time.perf_counter()
code = 1
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('/model', local_files_only=True,
                                       trust_remote_code=False)
    record['transformers_config_recognized'] = True
except Exception as exc:
    record.update(transformers_config_recognized=False,
                  config_exception_type=type(exc).__name__,
                  config_exception_message=str(exc))
try:
    from vllm import LLM
    from vllm.model_executor.models import ModelRegistry
    supported = ModelRegistry.get_supported_archs()
    record['target_architecture_registered'] = 'DeepseekV41ForCausalLM' in supported
    record['supported_deepseek_architectures'] = sorted(a for a in supported if 'deepseek' in a.lower())
    llm = LLM(**settings)
    record['engine_started'] = True
    code = 0
except Exception as exc:
    record.update(engine_started=False, exception_type=type(exc).__name__,
                  exception_message=str(exc))
    traceback.print_exc()
finally:
    record['elapsed_s'] = time.perf_counter() - start
    record['exit_code'] = code
    Path('/evidence/startup-result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
sys.exit(code)
