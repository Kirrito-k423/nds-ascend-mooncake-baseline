#!/usr/bin/env bash
set -eu
: "${NDS_PROJECT_DIR:?set the remote MAC-13 project directory}"
NDS_IMAGE="${NDS_IMAGE:-nds-vllm-source:kv-import-fix-20261012}"
NDS_CONTAINER=nds-mac13-v41-preflight
cd "$SHW_RESULTS_DIR"
cp "$NDS_PROJECT_DIR"/device_smoke.py "$NDS_PROJECT_DIR"/startup_probe.py .
npu-smi info > npu-before.txt
python3 - <<'PY'
import re
from pathlib import Path
t=Path('npu-before.txt').read_text()
idle={int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)\b',t)}
assert set(range(8)) <= idle, 'Selected NPU occupied; abort before any compute'
PY
npu-smi info -m > device-map.txt
npu-smi info -l > topology.txt
for i in 0 1 2 3 4 5 6 7; do
 npu-smi info -t current-fault-event -i "$i" > "fault-rank$i.txt" 2>&1 || true
done
docker image inspect "$NDS_IMAGE" --format '{{.Id}}' > image-id.txt
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-before.txt
python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(('127.0.0.1', 18213))
PY
cleanup() {
 docker rm -f "$NDS_CONTAINER" >/dev/null 2>&1 || true
 npu-smi info > npu-after.txt
 docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-after.txt
}
trap cleanup EXIT
set +e
timeout -k 20 420 docker run --rm --name "$NDS_CONTAINER" --privileged \
 --network host --security-opt label=disable --shm-size 2g --cpus 32 --memory 32g \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /etc/hccn.conf:/etc/hccn.conf:ro \
 -v "$SHW_RESULTS_DIR:/evidence" -v "$NDS_PROJECT_DIR/model:/model:ro" \
 -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
 -e ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' -e OMP_NUM_THREADS=2 \
 -e VLLM_VERSION=0.23.0 \
 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 "$NDS_IMAGE" bash -lc '
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /usr/local/Ascend/nnal/atb/set_env.sh
 export PATH=/opt/nds-venv/bin:$PATH
 python - <<"PY"
import json,importlib.util,subprocess
from pathlib import Path
out={}
for name in ("vllm","vllm_ascend"):
 spec=importlib.util.find_spec(name)
 root=Path(spec.origin).parent
 r=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],text=True,capture_output=True)
 out[name]={"package_path":str(root),"git_commit":r.stdout.strip() if r.returncode==0 else None}
Path("/evidence/source-revisions.json").write_text(json.dumps(out,indent=2))
for path in Path("/usr/local/Ascend").glob("*/latest/version.cfg"):
 Path("/evidence/"+path.parts[-3]+"-version.cfg").write_text(path.read_text())
PY
 timeout -k 10 180 python -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --rdzv-backend=static --master-addr=127.0.0.1 --master-port=18213 /evidence/device_smoke.py > /evidence/smoke.log 2>&1
 smoke_rc=$?
 echo "$smoke_rc" > /evidence/smoke-exit-code.txt
 if test "$smoke_rc" != 0; then exit "$smoke_rc"; fi
 timeout -k 10 180 python /evidence/startup_probe.py > /evidence/startup.log 2>&1
 startup_rc=$?
 echo "$startup_rc" > /evidence/startup-exit-code.txt
 exit "$startup_rc"
 '
NDS_RC=$?
echo "$NDS_RC" > container-exit-code.txt
exit "$NDS_RC"
