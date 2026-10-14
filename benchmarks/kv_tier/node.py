"""Bounded node controller. A foreground owner collects and stops both nodes."""
import base64
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BACKEND = '/src/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py'


def http(base, path, body=None, timeout=60):
    req = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as response: return response.read()


class Node:
    def __init__(self, role):
        self.role = role
        self.local = os.environ['NDS_LOCAL_HOST']
        self.source = os.environ['NDS_SOURCE_HOST']
        self.out = Path(os.environ['SHW_RESULTS_DIR'])
        self.project = Path(os.environ['NDS_PROJECT_DIR'])
        self.image = os.environ.get('NDS_IMAGE', 'nds-vllm-source:kv-import-fix-20261012')
        self.name = 'nds-mac16-'+role
        self.proc = None
        self.mode = None
        self.log = None
        self.stop_event = threading.Event()
        self.owned_offload_dirs = set()
        self.report = {'role': role, 'host': self.local, 'source': self.source, 'launches': []}
        self.persist()

    def persist(self):
        (self.out/'node.json').write_text(json.dumps(self.report, indent=2))
        print(json.dumps({'node':self.role,'launches':self.report['launches']}),flush=True)

    def snapshot(self):
        value = {'monotonic_ns': time.perf_counter_ns(), 'wall_time_ns': time.time_ns(),
                 'host': self.local, 'disk': {}, 'net': {}}
        for p in Path('/sys/block').glob('nvme*n1/stat'):
            fields = [int(x) for x in p.read_text().split()]
            value['disk'][p.parent.name] = {'read_bytes': fields[2]*512, 'write_bytes': fields[6]*512,
                                           'raw': fields}
        for p in Path('/sys/class/net').glob('*/statistics'):
            value['net'][p.parent.name] = {k: int((p/k).read_text()) for k in ['rx_bytes','tx_bytes']}
        value['cpu_stat'] = Path('/proc/stat').read_text().splitlines()[0]
        value['meminfo'] = Path('/proc/meminfo').read_text()
        value['npu'] = subprocess.run(['npu-smi','info'],capture_output=True,text=True).stdout
        pid_query = subprocess.run(['docker','inspect','--format','{{.State.Pid}}',self.name],
                                   capture_output=True,text=True)
        if pid_query.returncode == 0:
            try:
                pid = int(pid_query.stdout.strip())
                groups = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
                group = next(x.split('::',1)[1] for x in groups if x.startswith('0::'))
                folder = Path('/sys/fs/cgroup')/group.lstrip('/')
                value['container_io_stat'] = (folder/'io.stat').read_text()
                value['container_cpu_stat'] = (folder/'cpu.stat').read_text()
            except (OSError,ValueError,StopIteration) as e:
                value['container_counter_error'] = str(e)
        try: value['metrics'] = http('http://127.0.0.1:18196','/metrics').decode()
        except Exception as e: value['metrics_error'] = str(e)
        return value

    def start(self, mode):
        assert self.proc is None, 'A server is already running'
        for port in [18196, 18197]+([50296,50297] if self.role == 'source' else []):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                try: sock.bind(('0.0.0.0',port))
                except OSError as e: raise RuntimeError(f'Port {port} unavailable: {e}') from e
        self.mode = mode
        prefix = self.out/mode
        prefix.mkdir()
        # The destination controller may wait while the source model loads.
        # Recheck at the actual launch, rather than trusting the earlier gate.
        npu = subprocess.check_output(['npu-smi','info'])
        (prefix/'pre-start-npu.txt').write_bytes(npu)
        subprocess.run(['python3',str(self.project/'idle_gate.py'),str(prefix/'pre-start-npu.txt'),
                        os.environ.get('NDS_VISIBLE_DEVICES','0,1,2,3')],cwd=prefix,check=True)
        self.log = prefix/'server.log'
        offload = self.project/('offload-'+mode)
        offload.mkdir(exist_ok=True)
        assert not list(offload.iterdir()), 'Require an empty owned offload directory'
        self.owned_offload_dirs.add(offload)
        config = {'metadata_server':'P2PHANDSHAKE','protocol':'ascend','device_name':'',
                  'master_server_address':self.source+':50296',
                  'global_segment_size':100*1024**3 if self.role == 'source' else 0,
                  'local_buffer_size':1024**3,'preferred_segment':True,'prefer_alloc_in_same_node':True,
                  'enable_ssd_offload':mode == 'disk' and self.role == 'source',
                  'ssd_offload_path':'/offload'}
        (prefix/'mooncake.json').write_text(json.dumps(config,indent=2))
        base = subprocess.check_output(['docker','run','--rm','--entrypoint','cat',self.image,BACKEND])
        (prefix/'backend.original.py').write_bytes(base)
        subprocess.run(['python3', str(self.project/'instrument.py'), str(prefix/'backend.original.py'),
                        str(prefix/'backend.instrumented.py')],check=True)
        kv = {'kv_connector':'MooncakeConnectorStoreV1','kv_role':'kv_both',
              'kv_connector_extra_config':{'use_layerwise':False,'mooncake_rpc_port':'0',
                                            'load_async':True,'register_buffer':True}}
        server = ['/opt/nds-venv/bin/python','-m','vllm.entrypoints.openai.api_server',
                  '--model','/model','--served-model-name','qwen','--load-format','dummy','--dtype','bfloat16',
                  '--max-model-len','25600','--tensor-parallel-size','4','--host','0.0.0.0','--port','18196',
                  '--max-num-batched-tokens','4096','--max-num-seqs','32','--gpu-memory-utilization','0.9',
                  '--seed','42','--kv-transfer-config',json.dumps(kv)]
        # A5 already measured in graph mode. Keep the model and decode path fixed.
        shell = 'set -e; source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; export PATH=/opt/nds-venv/bin:$PATH; '
        if self.role == 'source':
            shell += 'mooncake_master --port=50296 --metrics_port=50297 --default_kv_lease_ttl=5000 --client_ttl=120 '
            if mode == 'disk': shell += '--enable_offload=true '
            shell += '> /evidence/'+mode+'/master.log 2>&1 & '
        shell += '/opt/nds-venv/bin/python /task/wait_master.py; '
        shell += '/opt/nds-venv/bin/python /task/store_agent.py > /evidence/'+mode+'/agent.log 2>&1 & '
        shell += 'exec '+shlex.join(server)
        env = {'ASCEND_RT_VISIBLE_DEVICES':os.environ.get('NDS_VISIBLE_DEVICES','0,1,2,3'),'ASCEND_LOCAL_COMM_RES':'{"version":"1.3"}',
               'OMP_NUM_THREADS':'4','VLLM_VERSION':'0.23.0','PYTHONHASHSEED':'0',
               'VLLM_HOST_IP':self.local,'MOONCAKE_CONFIG_PATH':'/evidence/'+mode+'/mooncake.json',
               'VLLM_LOGGING_LEVEL':'DEBUG','VLLM_SERVER_DEV_MODE':'1',
               'NDS_LOCAL_HOST':self.local,'NDS_MASTER':self.source+':50296','NDS_AGENT_PORT':'18197',
               'NDS_OWN_FILE_COLD':os.environ.get('NDS_OWN_FILE_COLD','1'),
               'NDS_WORKER_CLEAR':'1' if mode=='disk' and self.role=='source' else '0',
               'MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES':str(1024**3),
               'MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES':str(64*1024**3),
               'MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE':str(64*1024**3),
               'MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES':str(256*1024**2),
               'MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT':os.environ.get('NDS_OFFLOAD_BUCKET_KEYS_LIMIT','1'),
               'MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS':'1',
               'MOONCAKE_OFFLOAD_FILE_STORAGE_PATH':'/offload',
               'MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR':'bucket_storage_backend',
               'MOONCAKE_OFFLOAD_USE_URING':'false'}
        cmd = ['docker','run','--rm','--name',self.name,'--privileged','--network','host',
               '--security-opt','label=disable','--shm-size','16g','--cpus','32','--memory','800g']
        for a,b in [('/usr/local/Ascend/driver','/usr/local/Ascend/driver:ro'),
                    ('/usr/local/dcmi','/usr/local/dcmi:ro'),
                    ('/usr/local/sbin/npu-smi','/usr/local/sbin/npu-smi:ro'),
                    ('/etc/ascend_install.info','/etc/ascend_install.info:ro'),('/etc/hccn.conf','/etc/hccn.conf:ro'),
                    (str(self.out),'/evidence'),(str(self.project),'/task:ro'),
                    (os.environ['NDS_MODEL_DIR'],'/model:ro'),(str(offload),'/offload'),
                    (str(prefix/'backend.instrumented.py'),BACKEND+':ro')]: cmd += ['-v',a+':'+b]
        for k,v in env.items(): cmd += ['-e',k+'='+v]
        cmd += [self.image,'bash','-lc',shell]
        (prefix/'launch.json').write_text(json.dumps({'command':cmd,'config':config},indent=2))
        self.report['launches'].append({'mode':mode,'begin_wall_ns':time.time_ns()});self.persist()
        self.proc = subprocess.Popen(cmd,stdout=self.log.open('w'),stderr=subprocess.STDOUT)
        deadline = time.monotonic()+1800
        while time.monotonic() < deadline:
            if self.proc.poll() is not None: raise RuntimeError('vLLM exited '+str(self.proc.returncode))
            with self.log.open('rb') as f:
                f.seek(max(0,self.log.stat().st_size-65536))
                recent=f.read().decode(errors='replace')
            if 'RuntimeError: TransferEngine initialization failed' in recent:
                raise RuntimeError('Fatal TransferEngine initialization; preserve server.log and stop owned container')
            try:
                http('http://127.0.0.1:18196','/health',timeout=2)
                http('http://127.0.0.1:18197','/',timeout=2)
                self.report['launches'][-1]['ready_wall_ns'] = time.time_ns();self.persist()
                return {'ready':True,'mode':mode}
            except OSError: time.sleep(2)
        raise TimeoutError('Server readiness exceeded 1800s')

    def stop(self):
        if self.proc is not None:
            subprocess.run(['docker','stop','--time','15',self.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try: self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired: self.proc.terminate(); self.proc.wait(timeout=10)
            self.report['launches'][-1]['exit_after_stop'] = self.proc.returncode
            self.proc = None;self.persist()
        return {'stopped':True}

    def cleanup_offload(self):
        check = subprocess.run(['docker','ps','--format','{{.Names}}'],capture_output=True,text=True)
        assert check.returncode == 0 and self.name not in check.stdout.splitlines(), 'Container still exists'
        cleaned = []
        for p in self.owned_offload_dirs:
            assert p.parent == self.project and p.name in ['offload-memory','offload-disk']
            if p.exists(): shutil.rmtree(p)
            cleaned.append(p.name)
        (self.out/'offload-cleanup.json').write_text(json.dumps({'removed':sorted(cleaned)}))

    def dispatch(self, path, body):
        if path == '/start': return self.start(body['mode'])
        if path == '/stop': return self.stop()
        if path == '/shutdown': self.stop_event.set();return {'shutdown':True}
        if path == '/snapshot': return self.snapshot()
        if path == '/offset': return {'offset':self.log.stat().st_size}
        if path == '/log':
            with self.log.open('rb') as f: f.seek(body['offset']);return {'text':f.read().decode(errors='replace')}
        if path == '/agent': return json.loads(http('http://127.0.0.1:18197',body['path'],body['body'],timeout=120))
        raise ValueError('Unknown node action')


def serve(node):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result = node.dispatch(self.path,body);code=200
            except Exception:
                result={'error':traceback.format_exc()};code=500
            self.send_response(code);self.end_headers();self.wfile.write(json.dumps(result).encode())
        def do_GET(self):
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(node.report).encode())
        def log_message(self,*args): pass
    server = ThreadingHTTPServer(('0.0.0.0',18198),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        if not node.stop_event.wait(5400): raise TimeoutError('Controller exceeded 90 minute bound')
    finally:
        node.stop();node.cleanup_offload();server.shutdown();server.server_close()
