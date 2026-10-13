"""Tracked two-host 72B test. The source coordinates both owned services."""
import ast, concurrent.futures, http.server, json, os, pathlib, re, shlex, signal, socket, subprocess, sys, threading, time, traceback, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROLE=sys.argv[1]
SOURCE=os.environ['NDS_SOURCE_HOST'];DEST=os.environ['NDS_DESTINATION_HOST']
if SOURCE == DEST: raise SystemExit('Source and destination must be distinct hosts')
LOCAL=SOURCE if ROLE=='source' else DEST
BASE='http://127.0.0.1:18172';REMOTE='http://'+DEST+':18172';CONTROL='http://'+DEST+':18174'
OUT=pathlib.Path(os.environ['SHW_RESULTS_DIR'])
PROJECT=pathlib.Path(os.environ['NDS_PROJECT_DIR'])
if not PROJECT.is_absolute(): raise SystemExit('NDS_PROJECT_DIR must be an absolute remote directory')
IMAGE=os.environ.get('NDS_INFERENCE_IMAGE','nds-vllm-source:kv-import-fix-20261012');BENCH='nds-vllm-aisbench:typical72b-20261012'
NAME='nds-mac3-qwen72b-cross-'+ROLE
DEVICES=os.environ.get('NDS_VISIBLE_DEVICES','0,1,2,3')
children=[];stop=threading.Event();control_server=None
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
report={'scenario':'Qwen2.5-72B dummy BF16 TP4 PD-colocated two physical A5 hosts',
 'role':ROLE,'machine':os.environ.get('NDS_SOURCE_LABEL','source') if ROLE=='source' else os.environ.get('NDS_DESTINATION_LABEL','destination'),'devices':[int(i) for i in DEVICES.split(',')],
 'source_host':SOURCE,'destination_host':DEST,'warning_allowed':True,'global_segment_bytes_per_rank':107374182400,
 'input_tokens':1024,'output_tokens':10,'requests':100,'concurrency':25,'phases':[],
 'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
report['inference_image']=IMAGE

def save(): (OUT/'cross-node-result.json').write_text(json.dumps(report,indent=2))
def spawn(cmd,log):
    p=subprocess.Popen(cmd,stdout=open(OUT/log,'w'),stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p);return p
def http(base,path,post=False,body=None,timeout=30):
    req=urllib.request.Request(base+path,method='POST' if post else 'GET',data=body,
          headers={'Content-Type':'application/json'} if body is not None else {})
    with OPENER.open(req,timeout=timeout) as response:return response.read()
def reset(base):http(base,'/reset_prefix_cache?reset_external=false',post=True)
def bench_command(args):
    return ['docker','run','--rm','--network','host','--cpus','16','--memory','32g',
      '-v',str(OUT)+':/evidence','-v',str(PROJECT/'Qwen2.5-72B-Instruct')+':/model:ro',
      '-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','PYTHONHASHSEED=0','-e','VLLM_VERSION=0.23.0',
      '-e','HTTP_PROXY=','-e','HTTPS_PROXY=','-e','ALL_PROXY=',
      '-e','http_proxy=','-e','https_proxy=','-e','all_proxy=',
      '-e','NO_PROXY='+SOURCE+','+DEST+',127.0.0.1,localhost',
      BENCH,'/opt/nds-venv/bin/python',*args]
def bench(dataset,label,base=BASE):
    host='127.0.0.1' if base==BASE else DEST
    cfg=(PROJECT/'aisbench_config.template.py').read_text().replace('__DATASET__',dataset).replace("host_ip='127.0.0.1'","host_ip='"+host+"'")
    (OUT/(label+'.py')).write_text(cfg)
    (OUT/(label+'-metrics-before.txt')).write_bytes(http(base,'/metrics'))
    offset=(OUT/'server.log').stat().st_size if base==BASE else json.loads(http(CONTROL,'/offset'))['offset']
    begin=time.monotonic()
    p=spawn(bench_command(['-m','ais_bench.benchmark.cli.main','/evidence/'+label+'.py',
       '--summarizer','default_perf','--mode','perf','--num-warmups','0','--work-dir','/evidence/aisbench/'+label,'--debug']),label+'-client.log')
    rc=p.wait(timeout=600);time.sleep(2)
    (OUT/(label+'-metrics-after.txt')).write_bytes(http(base,'/metrics'))
    if base==BASE:
        with open(OUT/'server.log','rb') as log:log.seek(offset);excerpt=log.read()
    else:excerpt=http(CONTROL,'/log?offset='+str(offset),timeout=60)
    (OUT/(label+'-server.log')).write_bytes(excerpt)
    (OUT/(label+'-master-metrics.txt')).write_bytes(http('http://127.0.0.1:50273','/metrics'))
    report['phases'].append({'label':label,'dataset':dataset,'server':host,'exit_code':rc,'elapsed_s':time.monotonic()-begin});save()
    if rc:raise RuntimeError('AISBench failed: '+label+' exit '+str(rc))
def audit(label):
    text=(OUT/(label+'-server.log')).read_text()
    keys=sorted({key for match in re.finditer(r'MooncakeBackend.get enter keys=\d+ sample_keys=(\[[^\n]*?\])',text)
                 for key in ast.literal_eval(match[1])})
    assert keys, 'No replica keys in '+label
    (OUT/(label+'-keys.json')).write_text(json.dumps(keys))
    args=['/opt/nds-venv/bin/python','/evidence/replica_audit.py',
          '/evidence/'+label+'-keys.json','/evidence/'+label+'-replicas.json',SOURCE]
    shell='source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; '+shlex.join(args)
    cmd=['docker','exec',NAME,'bash','-lc',shell]
    p=spawn(cmd,label+'-replica-audit.log')
    if p.wait(timeout=120):raise RuntimeError('Source DRAM ownership audit failed: '+label)
def warm_shapes(base,label,dataset):
    rows=[json.loads(line) for line in (OUT/(dataset+'.jsonl')).read_text().splitlines()]
    results=[]
    for size in range(1,26):
        reset(base);barrier=threading.Barrier(size);begin=time.monotonic()
        def one(i):
            body=json.dumps({'model':'qwen','messages':[{'role':'user','content':rows[i]['question']}],
                             'max_tokens':10,'ignore_eos':True,'temperature':0}).encode()
            barrier.wait(timeout=30)
            usage=json.loads(http(base,'/v1/chat/completions',True,body,180))['usage']
            assert usage['prompt_tokens']==1024 and usage['completion_tokens']==10,usage
        with concurrent.futures.ThreadPoolExecutor(max_workers=size) as pool:list(pool.map(one,range(size)))
        results.append({'batch_size':size,'requests':size,'elapsed_s':time.monotonic()-begin})
    report[label]=results;save()

try:
    for name in ['replica_audit.py','store_replica_utils.py']:
        (OUT/name).write_bytes((PROJECT/name).read_bytes())
    for port in ([18172,50272,50273] if ROLE=='source' else [18172,18174]):
        with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',port))
    if ROLE=='destination':
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path=self.path.split('?')[0]
                if path=='/offset':body=json.dumps({'offset':(OUT/'server.log').stat().st_size}).encode()
                elif path=='/log':
                    offset=int(self.path.split('offset=')[1])
                    with open(OUT/'server.log','rb') as log:log.seek(offset);body=log.read()
                else:body=json.dumps(report).encode()
                self.send_response(200);self.end_headers();self.wfile.write(body)
            def do_POST(self):
                if self.path!='/stop':self.send_error(404);return
                self.send_response(200);self.end_headers();stop.set()
            def log_message(self,*args):pass
        control_server=ThreadingHTTPServer(('0.0.0.0',18174),Handler)
        threading.Thread(target=control_server.serve_forever,daemon=True).start()
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline and not stop.is_set():
            try:
                with socket.create_connection((SOURCE,50272),timeout=1):break
            except OSError:time.sleep(2)
        else:raise RuntimeError('Source Master readiness timeout or stop requested')
    mooncake={'metadata_server':'P2PHANDSHAKE','protocol':'ascend','device_name':'',
      'master_server_address':SOURCE+':50272','global_segment_size':107374182400,
      'preferred_segment':True,'prefer_alloc_in_same_node':True,'enable_ssd_offload':False}
    (OUT/'mooncake.json').write_text(json.dumps(mooncake,indent=2))
    mounts=['-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
      '-v','/usr/local/dcmi:/usr/local/dcmi:ro','-v','/usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro',
      '-v','/etc/ascend_install.info:/etc/ascend_install.info:ro','-v','/etc/hccn.conf:/etc/hccn.conf:ro',
      '-v',str(OUT)+':/evidence','-v',str(PROJECT/'Qwen2.5-72B-Instruct')+':/model:ro']
    env=['-e','ASCEND_RT_VISIBLE_DEVICES='+DEVICES,'-e','ASCEND_LOCAL_COMM_RES={"version":"1.3"}',
      '-e','OMP_NUM_THREADS=4','-e','VLLM_VERSION=0.23.0','-e','PYTHONHASHSEED=0',
      '-e','VLLM_HOST_IP='+LOCAL,'-e','MOONCAKE_CONFIG_PATH=/evidence/mooncake.json',
      '-e','VLLM_LOGGING_LEVEL=DEBUG','-e','VLLM_SERVER_DEV_MODE=1']
    kv={'kv_connector':'MooncakeConnectorStoreV1','kv_role':'kv_both',
      'kv_connector_extra_config':{'use_layerwise':False,'mooncake_rpc_port':'0','load_async':True,'register_buffer':True}}
    server=['/opt/nds-venv/bin/python','-m','vllm.entrypoints.openai.api_server',
      '--model','/model','--served-model-name','qwen','--load-format','dummy','--dtype','bfloat16',
      '--max-model-len','25600','--tensor-parallel-size','4','--host','0.0.0.0','--port','18172',
      '--max-num-batched-tokens','4096','--max-num-seqs','32','--gpu-memory-utilization','0.9',
      '--seed','42','--kv-transfer-config',json.dumps(kv)]
    report['enforce_eager']=os.environ.get('NDS_ENFORCE_EAGER')=='1'
    if report['enforce_eager']:server.append('--enforce-eager')
    shell='source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; export PATH=/opt/nds-venv/bin:$PATH; '
    if ROLE=='source':shell+='mooncake_master --port 50272 --metrics_port 50273 --default_kv_lease_ttl 11000 --client_ttl 120 > /evidence/master.log 2>&1 & '
    shell+='exec '+shlex.join(server)
    cmd=['docker','run','--rm','--name',NAME,'--privileged','--network','host','--security-opt','label=disable',
         '--shm-size','16g','--cpus','32','--memory','800g',*mounts,*env,IMAGE,'bash','-lc',shell]
    (OUT/'launch.json').write_text(json.dumps({'docker_command':cmd,'server_command':server},indent=2))
    p=spawn(cmd,'server.log');save();deadline=time.monotonic()+1800
    while time.monotonic()<deadline and not stop.is_set():
        if p.poll() is not None:raise RuntimeError('vLLM exited '+str(p.returncode))
        try:http(BASE,'/health');break
        except OSError:time.sleep(2)
    else:raise RuntimeError('vLLM readiness timeout or stop requested')
    report['server_ready_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()
    if ROLE=='destination':
        while not stop.wait(2):
            if p.poll() is not None:raise RuntimeError('Destination vLLM exited early')
            if (OUT/'stop.request').exists():break
    else:
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            try:http(REMOTE,'/health');break
            except OSError:time.sleep(2)
        else:raise RuntimeError('Destination vLLM readiness timeout')
        remote_info=json.loads(http(CONTROL,'/'))
        report['destination_devices']=remote_info['devices'];save()
        p=spawn(bench_command(['/evidence/make_dataset_cross.py']),'dataset.log')
        if p.wait(timeout=240):raise RuntimeError('Dataset generation failed')
        for base,node in [(BASE,'source'),(REMOTE,'destination')]:
            smoke={'model':'qwen','messages':[{'role':'user','content':'Hello'}],
                   'max_tokens':10,'ignore_eos':True,'temperature':0}
            result=json.loads(http(base,'/v1/chat/completions',True,json.dumps(smoke).encode(),180))
            assert result['usage']['completion_tokens']==10,result
            (OUT/(node+'-smoke-response.json')).write_text(json.dumps(result,indent=2))
            warm='warm' if node=='source' else 'warm-destination'
            bench(warm,node+'-warmup-100',base);reset(base)
            bench(warm,node+'-warmup-dram-100',base);warm_shapes(base,node+'-batch-shapes',warm)
        for repeat in range(3):
            dataset='A'+str(repeat)
            reset(REMOTE);bench('C'+str(repeat),'cold-destination-'+str(repeat),REMOTE)
            reset(BASE);bench(dataset,'cold-source-'+str(repeat))
            reset(BASE);bench(dataset,'prime-source-'+str(repeat))
            reset(BASE);bench(dataset,'local-source-'+str(repeat));audit('local-source-'+str(repeat))
            reset(REMOTE);bench(dataset,'remote-destination-'+str(repeat),REMOTE);audit('remote-destination-'+str(repeat))
        report['sampling_finished']=True
except Exception:
    report['error']=traceback.format_exc();raise
finally:
    save()
    if ROLE=='source':
        try:http(CONTROL,'/stop',post=True,timeout=10)
        except Exception:pass
    subprocess.run(['docker','stop','--time','15',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for child in reversed(children):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if control_server is not None:control_server.shutdown();control_server.server_close()
    report['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()
