#!/usr/bin/env bash
set -eu
: "${NDS_ROLE:?source or destination}"
: "${NDS_SOURCE_HOST:?physical source address}"
: "${NDS_DESTINATION_HOST:?different physical destination address}"
: "${NDS_LOCAL_HOST:?this physical node address}"
: "${NDS_PROJECT_DIR:?private toolkit directory}"
: "${NDS_MODEL_DIR:?complete model config and tokenizer directory}"
: "${NDS_TOOLKIT_SHA256:?hash of toolkit .py files}"
: "${SHW_RESULTS_DIR:?task controller result directory}"
NDS_VISIBLE_DEVICES="${NDS_VISIBLE_DEVICES:-0,1,2,3}"
export NDS_VISIBLE_DEVICES
cd "$SHW_RESULTS_DIR"
echo "results_dir=$SHW_RESULTS_DIR"
python3 - <<'PY'
import hashlib,os,pathlib
root=pathlib.Path(os.environ['NDS_PROJECT_DIR'])
lines=[p.name+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))]
assert hashlib.sha256('\n'.join(lines).encode()).hexdigest()==os.environ['NDS_TOOLKIT_SHA256'],'Stale toolkit; redeploy the reviewed files'
assert os.environ['NDS_SOURCE_HOST']!=os.environ['NDS_DESTINATION_HOST']
assert os.environ['NDS_LOCAL_HOST']==os.environ['NDS_SOURCE_HOST' if os.environ['NDS_ROLE']=='source' else 'NDS_DESTINATION_HOST']
PY
npu-smi info > before.txt
python3 "$NDS_PROJECT_DIR/idle_gate.py" before.txt "$NDS_VISIBLE_DEVICES"
findmnt -J -T "$NDS_PROJECT_DIR" > offload-mount.json
cat /proc/meminfo > before-meminfo.txt
trap 'docker rm -f "nds-mac16-$NDS_ROLE" >/dev/null 2>&1 || true; npu-smi info > after.txt' EXIT
timeout -k 30 5400 python3 "$NDS_PROJECT_DIR/run.py" "$NDS_ROLE"
