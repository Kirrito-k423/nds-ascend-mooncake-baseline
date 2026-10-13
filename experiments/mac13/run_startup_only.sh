#!/usr/bin/env bash
set -eu
: "${NDS_PROJECT_DIR:?set remote MAC-13 project directory}"
cd "$SHW_RESULTS_DIR"
cp "$NDS_PROJECT_DIR/startup_probe.py" .
npu-smi info > npu-before.txt
python3 - <<'PY'
import re
from pathlib import Path
t=Path('npu-before.txt').read_text()
assert set(range(8)) <= {int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)\b',t)}
PY
NDS_CONTAINER=nds-mac13-v41-startup
cleanup() {
 docker rm -f "$NDS_CONTAINER" >/dev/null 2>&1 || true
 npu-smi info > npu-after.txt
 docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-after.txt
}
trap cleanup EXIT
set +e
timeout -k 20 180 docker run --rm --name "$NDS_CONTAINER" --privileged \
 --network host --security-opt label=disable --shm-size 2g --cpus 32 --memory 32g \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /etc/hccn.conf:/etc/hccn.conf:ro \
 -v "$SHW_RESULTS_DIR:/evidence" -v "$NDS_PROJECT_DIR/model:/model:ro" \
 -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
 -e ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' -e OMP_NUM_THREADS=2 \
 -e VLLM_VERSION=0.23.0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 nds-vllm-source:kv-import-fix-20261012 bash -lc '
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /usr/local/Ascend/nnal/atb/set_env.sh
 /opt/nds-venv/bin/python /evidence/startup_probe.py > /evidence/startup.log 2>&1
 '
NDS_RC=$?
echo "$NDS_RC" > container-exit-code.txt
exit "$NDS_RC"
