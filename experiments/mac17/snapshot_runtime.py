"""Record versions and actual installed KV Pool files without environment dumps."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

root=Path('/evidence/runtime-snapshot');root.mkdir(exist_ok=True)
versions={'python':platform.python_version()}
for name in ['torch','torch-npu','vllm','vllm-ascend','transformers','tokenizers','mooncake-transfer-engine-npu']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
files=[]
for source in Path('/src/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store').rglob('*.py'):
    relative=source.relative_to('/src/vllm-ascend');target=root/relative
    target.parent.mkdir(parents=True,exist_ok=True);body=source.read_bytes();target.write_bytes(body)
    files.append({'path':str(relative),'sha256':hashlib.sha256(body).hexdigest()})
for base in ['/usr/local/Ascend/ascend-toolkit/latest','/usr/local/Ascend/nnal/atb/latest']:
    cfg=Path(base)/'version.cfg'
    if cfg.exists():versions[base]=cfg.read_text()
for name in ['/usr/local/Ascend/cann-9.1.0/share/info/runtime/version.info',
             '/usr/local/Ascend/driver/version.info']:
    cfg=Path(name)
    if cfg.exists():versions[name]=cfg.read_text()
versions['atb_resolved_path']=str(Path('/usr/local/Ascend/nnal/atb/latest').resolve())
(root/'manifest.json').write_text(json.dumps({'versions':versions,'files':files},indent=2))
print(json.dumps(versions))
