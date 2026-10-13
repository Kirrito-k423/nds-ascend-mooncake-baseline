"""Foreground DP/EP server harness with exact token counts and owned cleanup."""
import argparse
import json
import os
import socket
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--max-model-len',type=int,default=200032)
    p.add_argument('--lengths',default='8192,32768,100000,200000')
    p.add_argument('--dp-size',type=int,default=8)
    p.add_argument('--port',type=int,default=18218)
    p.add_argument('--pool',action='store_true')
    p.add_argument('--repeats',type=int,default=1)
    a=p.parse_args();root=Path('/evidence');base=f'http://127.0.0.1:{a.port}'
    with socket.socket() as s:s.bind(('127.0.0.1',a.port))
    command=['vllm','serve','/model','--served-model-name','mac17-v4flash',
        '--host','127.0.0.1','--port',str(a.port),'--load-format','dummy',
        '--dtype','bfloat16','--quantization','fp8','--tensor-parallel-size','1',
        '--data-parallel-size',str(a.dp_size),'--enable-expert-parallel',
        '--worker-cls','mac17_worker.AuditWorker','--max-model-len',str(a.max_model_len),
        '--max-num-seqs','1','--max-num-batched-tokens','2048',
        '--gpu-memory-utilization','0.9','--block-size','128',
        '--tokenizer-mode','deepseek_v4','--no-enable-prefix-caching',
        '--no-disable-hybrid-kv-cache-manager','--enforce-eager','--cpu-offload-gb','0',
        '--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':1,'enforce_eager':True})]
    master=None
    if a.pool:
        for port in (50217,50218):
            with socket.socket() as s:s.bind(('0.0.0.0',port))
        from vllm.utils.network_utils import get_ip
        host=get_ip();os.environ['VLLM_HOST_IP']=host
        cfg={'metadata_server':'P2PHANDSHAKE','protocol':'ascend','device_name':'',
             'master_server_address':host+':50217','global_segment_size':1073741824,
             'preferred_segment':True,'prefer_alloc_in_same_node':True,'enable_ssd_offload':False}
        (root/'mooncake.json').write_text(json.dumps(cfg,indent=2))
        os.environ['MOONCAKE_CONFIG_PATH']=str(root/'mooncake.json')
        os.environ['MAC17_POOL_BYTE_AUDIT']='1'
        os.environ['VLLM_LOGGING_LEVEL']='DEBUG'
        master=subprocess.Popen(['mooncake_master','--port','50217','--metrics_port','50218',
            '--default_kv_lease_ttl','11000','--client_ttl','120'],
            stdout=(root/'master.log').open('wb'),stderr=subprocess.STDOUT)
        for _ in range(30):
            if master.poll() is not None:raise RuntimeError('owned master exited before readiness')
            try:
                with socket.create_connection(('127.0.0.1',50217),timeout=1):break
            except OSError:time.sleep(1)
        else:
            master.terminate();master.wait(timeout=10)
            raise TimeoutError('owned master readiness timeout')
        command+=['--kv-transfer-config',json.dumps({'kv_connector':'AscendStoreConnector',
            'kv_role':'kv_both','kv_connector_extra_config':{'use_layerwise':False,
            'mooncake_rpc_port':'0','load_async':True,'register_buffer':True}})]
    (root/'server-command.json').write_text(json.dumps(command,indent=2))
    result={'engine_started':False,'requests':[],'full_structure_requested':True,
            'tp_size':1,'dp_size':a.dp_size,'ep_enabled':True,
            'dummy_business_accuracy_evaluated':False}
    log=(root/'server.log').open('wb');server=subprocess.Popen(command,stdout=log,stderr=log)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(path,body=None,timeout=30):
        data=json.dumps(body).encode() if body is not None else None
        req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=timeout) as r:
            data=r.read()
            return json.loads(data) if data else None
    start=time.monotonic();code=1
    try:
        while time.monotonic()-start<1200:
            if server.poll() is not None:raise RuntimeError(f'server exited {server.returncode} before readiness')
            if (root/'server.log').exists():
                with (root/'server.log').open('rb') as check:
                    check.seek(max(0,(root/'server.log').stat().st_size-262144))
                    if b'Engine core initialization failed' in check.read():
                        raise RuntimeError('server log confirms engine initialization failed')
            try:call('/health',timeout=2);break
            except (OSError,urllib.error.URLError):time.sleep(2)
        else:raise TimeoutError('engine startup exceeded 20 minutes')
        result.update(engine_started=True,startup_s=time.monotonic()-start)
        models=call('/v1/models');(root/'models.json').write_text(json.dumps(models,indent=2))
        encoded=call('/tokenize',{'model':'mac17-v4flash','prompt':' A short sequence for the capacity experiment.',
                                  'add_special_tokens':False})
        (root/'tokenized-unit.json').write_text(json.dumps(encoded,indent=2))
        unit=encoded['tokens'];assert unit,'empty tokenized unit'
        for attempt,length in enumerate(list(map(int,a.lengths.split(',')))*a.repeats):
            assert length+32<=a.max_model_len
            prompt=(unit*((length+len(unit)-1)//len(unit)))[:length]
            record={'attempt':attempt,'requested_input_tokens':length,'tokenized_input_tokens':len(prompt),
                    'actual_input_tokens':None,'actual_output_tokens':None,'passed':None,'attempted':True}
            result['requests'].append(record);(root/'result.json').write_text(json.dumps(result,indent=2))
            t=time.monotonic()
            response=call('/v1/completions',{'model':'mac17-v4flash','prompt':prompt,
                'temperature':0,'max_tokens':32,'min_tokens':32,'ignore_eos':True},timeout=1800)
            (root/f'response-{length}-{attempt}.json').write_text(json.dumps(response,indent=2))
            record.update(actual_input_tokens=response['usage']['prompt_tokens'],
                actual_output_tokens=response['usage']['completion_tokens'],elapsed_s=time.monotonic()-t,
                finish_reason=response['choices'][0]['finish_reason'])
            record['passed']=record['actual_input_tokens']==length and record['actual_output_tokens']>=32
            (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(record),flush=True)
            if not record['passed']:raise RuntimeError('request length/output criterion failed')
            if a.pool:
                # Snapshot connector metrics, then allow asynchronous puts to complete.
                with opener.open(base+'/metrics',timeout=30) as r:
                    (root/f'metrics-{attempt}.txt').write_bytes(r.read())
                time.sleep(3)
        code=0
    except Exception as exc:
        result.update(exception_type=type(exc).__name__,exception_message=str(exc))
        if isinstance(exc,urllib.error.HTTPError):
            (root/'http-error-body.txt').write_bytes(exc.read(1024*1024))
        traceback.print_exc()
    finally:
        if server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=30)
        log.close();result.update(exit_code=code,server_exit_code=server.returncode,
                                  elapsed_s=time.monotonic()-start)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        if master is not None:
            master.terminate()
            try:master.wait(timeout=15)
            except subprocess.TimeoutExpired:master.kill();master.wait(timeout=15)
    raise SystemExit(code)

if __name__=='__main__':main()
