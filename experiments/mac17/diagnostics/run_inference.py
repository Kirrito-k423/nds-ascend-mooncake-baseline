"""Full-target startup and ordered requests; stop on the first failure."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--max-model-len',type=int,default=200032)
    p.add_argument('--lengths',default='8192,32768,100000,200000')
    p.add_argument('--startup-only',action='store_true')
    a=p.parse_args();root=Path('/evidence')
    versions={'python':platform.python_version()}
    for package in ['torch','torch-npu','vllm','vllm-ascend','transformers','tokenizers','mooncake-transfer-engine']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    (root/'versions.json').write_text(json.dumps(versions,indent=2))

    settings=dict(model='/model',tokenizer='/model',load_format='dummy',dtype='bfloat16',
        quantization='fp8',tensor_parallel_size=8,data_parallel_size=1,enable_expert_parallel=True,
        worker_cls='mac17_worker.AuditWorker',max_model_len=a.max_model_len,
        max_num_seqs=1,max_num_batched_tokens=2048,gpu_memory_utilization=0.9,
        enable_prefix_caching=False,disable_hybrid_kv_cache_manager=False,
        enforce_eager=True,cpu_offload_gb=0,block_size=128,tokenizer_mode='deepseek_v4',
        kv_cache_dtype='auto',speculative_config={'method':'mtp','num_speculative_tokens':1,'enforce_eager':True})
    (root/'settings.json').write_text(json.dumps(settings,indent=2))
    status={'engine_started':False,'full_structure_requested':True,'real_weight_payload_downloaded':False,
            'dummy_business_accuracy_evaluated':False,'requests':[]}
    start=time.monotonic();code=1
    stop_sampling=threading.Event()
    def sample():
        with (root/'device-samples.jsonl').open('a') as f:
            while not stop_sampling.is_set():
                try:
                    r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=10)
                    f.write(json.dumps({'monotonic_s':time.monotonic(),'time_ns':time.time_ns(),
                                        'exit_code':r.returncode,'stdout':r.stdout})+'\n');f.flush()
                except Exception as exc:
                    f.write(json.dumps({'error_type':type(exc).__name__})+'\n');f.flush()
                stop_sampling.wait(5)
    sampler=threading.Thread(target=sample);sampler.start()
    try:
        from vllm import LLM,SamplingParams
        llm=LLM(**settings)
        status['engine_started']=True;status['startup_s']=time.monotonic()-start
        (root/'result.json').write_text(json.dumps(status,indent=2))
        if not a.startup_only:
            tokenizer=llm.get_tokenizer()
            text=' A short sequence for the capacity experiment.'
            unit=tokenizer.encode(text,add_special_tokens=False)
            assert unit, 'empty tokenizer result'
            for length in map(int,a.lengths.split(',')):
                assert length+32<=a.max_model_len,'prompt plus output exceeds engine limit'
                prompt=(unit*((length+len(unit)-1)//len(unit)))[:length]
                elapsed=time.monotonic()
                record={'requested_input_tokens':length,'tokenized_input_tokens':len(prompt),
                        'actual_input_tokens':None,'actual_output_tokens':None,
                        'attempted':True,'passed':None}
                status['requests'].append(record)
                (root/'result.json').write_text(json.dumps(status,indent=2))
                outputs=llm.generate([{'prompt_token_ids':prompt}],
                    SamplingParams(temperature=0,max_tokens=32,min_tokens=32,ignore_eos=True),use_tqdm=False)
                output=outputs[0];completion=output.outputs[0]
                record.update({'actual_input_tokens':len(output.prompt_token_ids),
                    'actual_output_tokens':len(completion.token_ids),'elapsed_s':time.monotonic()-elapsed,
                    'finish_reason':completion.finish_reason,'token_ids':completion.token_ids})
                record['passed']=record['actual_input_tokens']==length and record['actual_output_tokens']>=32
                (root/'result.json').write_text(json.dumps(status,indent=2))
                print(json.dumps(record),flush=True)
                if not record['passed']:raise RuntimeError('request length/output criterion failed')
        code=0
    except Exception as exc:
        status.update(exception_type=type(exc).__name__,exception_message=str(exc))
        traceback.print_exc()
    finally:
        stop_sampling.set();sampler.join(timeout=15)
        status.update(exit_code=code,elapsed_s=time.monotonic()-start)
        (root/'result.json').write_text(json.dumps(status,indent=2))
        print(json.dumps(status),flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
