#!/usr/bin/env bash
# Stage the supplied, verified metadata/toolkit only. Never starts a model.
set -euo pipefail
MAC16_BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAC16_MACHINE_ID="${1:?supply a5-b02-1 or the selected destination}"
MAC16_DESTINATION_ID="${2:-a5-b02-2}"
export MAC16_BUNDLE_ROOT MAC16_MACHINE_ID MAC16_DESTINATION_ID
cd "$MAC16_BUNDLE_ROOT"
sha256sum -c SHA256SUMS
python3 - <<'PY'
import json,os,shlex,shutil,subprocess
from pathlib import Path
bundle=Path(os.environ['MAC16_BUNDLE_ROOT'])
nodes=json.loads((bundle/'deploy/machines.json').read_text())['nodes']
mid=os.environ['MAC16_MACHINE_ID'];dest=os.environ['MAC16_DESTINATION_ID']
assert dest in ['a5-b02-2','a5-b02-3']
assert mid in ['a5-b02-1',dest]
expected='sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270'
image='nds-vllm-source:kv-import-fix-20261012'
actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
assert actual==expected,'Import the existing pinned image before staging; no rebuild/download fallback'
local_ips={a['local'] for r in json.loads(subprocess.check_output(['ip','-j','-4','address','show'],text=True)) for a in r.get('addr_info',[])}
assert nodes[mid]['data_ip'] in local_ips,'Wrong physical node'
target=Path('/home/t00906153/nds-mac16-b02-20261014')
assert not target.exists() and not target.is_symlink(),'Existing path: inspect ownership and manifest before updating'
target.mkdir(parents=True)
shutil.copytree(bundle/'payload/kv_tier',target/'kv_tier')
shutil.copytree(bundle/'payload/model',target/'model')
role='source' if mid=='a5-b02-1' else 'destination'
env={'NDS_ROLE':role,'NDS_SOURCE_HOST':nodes['a5-b02-1']['data_ip'],
     'NDS_DESTINATION_HOST':nodes[dest]['data_ip'],'NDS_LOCAL_HOST':nodes[mid]['data_ip'],
     'NDS_PROJECT_DIR':str(target/'kv_tier'),'NDS_MODEL_DIR':str(target/'model/Qwen2.5-72B-Instruct'),
     'NDS_TOOLKIT_SHA256':'56cedb68c9ba73d95613238215f0aa41a9c45bb9273098ec6d6fa94e0ce8076b',
     'NDS_VISIBLE_DEVICES':'0,1,2,3','NDS_MODES_JSON':'["disk"]',
     'NDS_CONFIGS_JSON':'[[1024,1]]'}
(target/'launch.env').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items()))
print('Metadata and toolkit staged; no NPU process started. Reservation and native gates remain required.')
PY
