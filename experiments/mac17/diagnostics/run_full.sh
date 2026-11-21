#!/usr/bin/env bash
set -eu
: "${NDS_PROJECT_DIR:?set independent remote project directory}"
NDS_IMAGE="${NDS_IMAGE:-nds-vllm-source:kv-import-fix-20261012}"
NDS_CONTAINER=nds-mac17-v4flash-full
cd "$SHW_RESULTS_DIR"
npu-smi info > npu-before.txt
python3 - <<'PY'
import re
from pathlib import Path
idle=set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)\b',Path('npu-before.txt').read_text())))
assert set(range(8))<=idle,'Selected NPU occupied; abort before compute'
PY
docker image inspect "$NDS_IMAGE" --format '{{.Id}}' > image-id.txt
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-before.txt
cleanup() {
 docker rm -f "$NDS_CONTAINER" >/dev/null 2>&1 || true
 npu-smi info > npu-after.txt
 docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-after.txt
}
trap cleanup EXIT
set +e
timeout -k 30 3600 docker run --rm --name "$NDS_CONTAINER" --privileged --network host \
 --security-opt label=disable --shm-size 16g --cpus 64 --memory 384g \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro \
 -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /etc/hccn.conf:/etc/hccn.conf:ro \
 -v "$SHW_RESULTS_DIR:/evidence" -v "$NDS_PROJECT_DIR/model:/model:ro" \
 -v "$NDS_PROJECT_DIR/scripts:/scripts:ro" \
 -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
 -e ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' -e OMP_NUM_THREADS=4 \
 -e VLLM_VERSION=0.23.0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e HCCL_CONNECT_TIMEOUT=120 -e HCCL_EXEC_TIMEOUT=120 \
 -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True -e MAC17_EVIDENCE=/evidence \
 "$NDS_IMAGE" bash -lc '
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /usr/local/Ascend/nnal/atb/set_env.sh
 export PATH=/opt/nds-venv/bin:$PATH
 export PYTHONPATH=/scripts:${PYTHONPATH:-}
 timeout -k 20 3300 python /scripts/run_inference.py > /evidence/inference.log 2>&1
 '
NDS_RC=$?
echo "$NDS_RC" > container-exit-code.txt
exit "$NDS_RC"
