"""Check staged bytes and model metadata on CPU; never initialize an NPU."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/t00906153/nds-mac16-b02-20261014')
IMAGE = 'nds-vllm-source:kv-import-fix-20261012'
IMAGE_ID = 'sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270'
manifest = json.loads((ROOT/'staging-manifest.json').read_text())
assert manifest['model_started'] is False
for name, expected in manifest['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
files = sorted((ROOT/'kv_tier').glob('*.py'))
lines = [p.name+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
kit = hashlib.sha256('\n'.join(lines).encode()).hexdigest()
assert kit == manifest.get('toolkit_sha256','56cedb68c9ba73d95613238215f0aa41a9c45bb9273098ec6d6fa94e0ce8076b')
probe = hashlib.sha256((ROOT/'sdk_tier_probe.py').read_bytes()).hexdigest()
assert probe == 'e5367b7dcbe5f3105fef6e14b085f420c48988e71a873283627b02436e95009b'
for p in files+[ROOT/'sdk_tier_probe.py']: ast.parse(p.read_text())
actual = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',IMAGE],text=True).strip()
assert actual == IMAGE_ID
mid = manifest['machine_id']; octet = {'a5-b02-1':21,'a5-b02-2':22,'a5-b02-3':23}[mid]
ips = {a['local'] for r in json.loads(subprocess.check_output(['ip','-j','-4','address','show'],text=True)) for a in r.get('addr_info',[])}
assert f'177.5.{octet}.1' in ips
container = 'mac16-b02-cpu-'+mid.rsplit('-',1)[1]+'-20261014'
code = '''
import importlib.metadata,json
from transformers import AutoConfig,AutoTokenizer
p='/model';c=AutoConfig.from_pretrained(p,local_files_only=True);t=AutoTokenizer.from_pretrained(p,local_files_only=True)
assert (c.num_hidden_layers,c.hidden_size,c.num_attention_heads,c.num_key_value_heads)==(80,8192,64,8)
s='MAC-16 tokenizer 校验';ids=t.encode(s,add_special_tokens=False);assert t.decode(ids)==s
versions={}
for n in ['torch','torch-npu','vllm','transformers','mooncake-transfer-engine']:
 try:versions[n]=importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:versions[n]=None
print(json.dumps({'config_layers':c.num_hidden_layers,'tokenizer_class':type(t).__name__,'tokenizer_smoke_tokens':len(ids),'tokenizer_roundtrip':True,'versions':versions}))
'''
command = ['docker','run','--rm','--name',container,'--network','none',
           '-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0',
           '-v',str(ROOT/'model/Qwen2.5-72B-Instruct')+':/model:ro','--entrypoint','python3',IMAGE,'-c',code]
try:
    runtime = subprocess.run(command,text=True,capture_output=True,timeout=120)
except subprocess.TimeoutExpired:
    subprocess.run(['docker','rm','-f',container],capture_output=True,check=False)
    raise
assert runtime.returncode == 0, runtime.stderr
cpu = json.loads(runtime.stdout.strip().splitlines()[-1])
result = {'machine_id':mid,'image_id':actual,'toolkit_sha256':kit,'sdk_probe_sha256':probe,
          'cpu_backend_autoload_disabled':True,
          'verified_file_count':len(manifest['files']),'cpu':cpu,
          'mount':json.loads(subprocess.check_output(['findmnt','-J','-T',str(ROOT)],text=True)),
          'model_npu_validated':False,'new_compute_hccl_validated':False,'container_removed':True}
assert subprocess.run(['docker','container','inspect',container],capture_output=True).returncode != 0
output = Path(os.environ['SHW_RESULTS_DIR'])/'staging-check.json'
output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False))
