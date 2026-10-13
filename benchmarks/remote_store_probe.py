"""Two physical hosts: source DRAM only, destination reads into NPU memory."""
import gc, hashlib, http.server, json, os, pathlib, socket, subprocess, sys, threading, time, traceback, urllib.request
import torch, torch_npu

OUT = pathlib.Path('/evidence')
ROLE, LOCAL, SOURCE = sys.argv[1:4]
report = {'role':ROLE,'local_host':LOCAL,'source_host':SOURCE,'protocol':'ascend',
          'local_comm_res':os.environ.get('ASCEND_LOCAL_COMM_RES'),'rows':[]}
store = None
master = None

def serialize(obj, depth=0):
    if obj is None or isinstance(obj,(str,int,float,bool)): return obj
    if isinstance(obj,(list,tuple)): return [serialize(v,depth+1) for v in obj]
    if isinstance(obj,dict): return {str(k):serialize(v,depth+1) for k,v in obj.items()}
    if depth>5: return str(obj)
    out={}
    for name in dir(obj):
        if name.startswith('_'):continue
        try: value=getattr(obj,name)
        except Exception:continue
        if not callable(value):out[name]=serialize(value,depth+1)
    return out or str(obj)

def replicas(key):
    rows=[]
    for desc in store.get_replica_desc(key):
        row={'status':str(desc.status),'memory_replica':desc.is_memory_replica()}
        if row['memory_replica']:row['memory']=serialize(desc.get_memory_descriptor())
        rows.append(row)
    return rows

def view(size):
    owner=torch.empty(size+2097152,dtype=torch.uint8,device='npu:0')
    offset=(-owner.data_ptr())%2097152
    return owner,owner[offset:offset+size]

def payload(size):
    return torch.arange(256,dtype=torch.uint8).repeat(size//256)

try:
    torch.set_num_threads(2);torch.npu.set_device(0)
    from mooncake.store import MooncakeDistributedStore
    if ROLE=='source':
        for port in [50276,50277,51172]:
            with socket.socket() as s:s.bind(('0.0.0.0',port))
        master=subprocess.Popen(['/opt/nds-venv/bin/mooncake_master','--port','50276','--metrics_port','50277'],
            stdout=open(OUT/'master.log','w'),stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                with socket.create_connection((SOURCE,50276),timeout=.2):break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('Master readiness timeout')
    store=MooncakeDistributedStore()
    segment=268435456 if ROLE=='source' else 0
    report['contributed_dram_bytes']=segment
    rc=store.setup(LOCAL,'P2PHANDSHAKE',segment,33554432,'ascend','',SOURCE+':50276')
    report['setup_return']=rc;assert rc==0,rc
    report['store_hostname']=store.get_hostname()
    if ROLE=='source':
        manifest=[]
        for mib in [1,8]:
            size=mib*1048576;host=payload(size);owner,src=view(size)
            src.copy_(host);torch.npu.synchronize()
            assert store.register_buffer(src.data_ptr(),size)==0
            key='nds-cross-probe-'+str(time.time_ns())+'-'+str(mib)
            rc=store.put_from(key,src.data_ptr(),size);assert rc==0,rc
            record={'key':key,'size':size,'sha256':hashlib.sha256(host.numpy().tobytes()).hexdigest(),
                    'replicas':replicas(key)}
            manifest.append(record)
            assert store.unregister_buffer(src.data_ptr())==0
        report['manifest']=manifest
        (OUT/'source-manifest.json').write_text(json.dumps(report,indent=2))
        done=threading.Event()
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body=json.dumps(report).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
            def do_POST(self):
                self.send_response(200);self.end_headers();done.set()
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('0.0.0.0',51172),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        if not done.wait(timeout=240):raise RuntimeError('Remote client completion timeout')
        server.shutdown();server.server_close();report['client_completed']=True
    else:
        with urllib.request.urlopen('http://'+SOURCE+':51172',timeout=10) as response:
            source=json.load(response)
        for record in source['manifest']:
            size=record['size'];owner,dst=view(size)
            assert store.register_buffer(dst.data_ptr(),size)==0
            desc=replicas(record['key'])
            assert desc and all(row['memory_replica'] for row in desc),desc
            assert SOURCE in json.dumps(desc),('source ownership not visible',desc)
            times=[]
            for i in range(8):
                dst.zero_();torch.npu.synchronize();start=time.perf_counter()
                rc=store.get_into(record['key'],dst.data_ptr(),size);torch.npu.synchronize()
                elapsed=time.perf_counter()-start;assert rc==size,(rc,size)
                actual=dst.cpu();assert torch.equal(actual,payload(size)),'Remote DRAM data mismatch'
                assert hashlib.sha256(actual.numpy().tobytes()).hexdigest()==record['sha256']
                if i>=3:times.append(elapsed*1000)
            report['rows'].append({'size':size,'verified_reads':8,'warmup_reads':3,
                'measured_ms':times,'source_replicas':desc,'data_exact':True})
            assert store.unregister_buffer(dst.data_ptr())==0
        report['remote_dram_reads_verified']=True
except Exception:
    report['error']=traceback.format_exc();raise
finally:
    (OUT/('remote-store-'+ROLE+'.json')).write_text(json.dumps(report,indent=2))
    if ROLE=='client':
        try:
            req=urllib.request.Request('http://'+SOURCE+':51172/done',method='POST')
            with urllib.request.urlopen(req,timeout=5) as response:response.read()
        except Exception:pass
    # Destroy the native client while torch_npu and the ACL runtime are alive.
    # The import must follow set_device; changing GC alone did not fix the
    # allocator abort in the first two runs.
    if store is not None:
        del store
        gc.collect()
    if master is not None:
        master.terminate()
        try:master.wait(timeout=5)
        except subprocess.TimeoutExpired:master.kill();master.wait()
print(json.dumps(report,indent=2),flush=True)
