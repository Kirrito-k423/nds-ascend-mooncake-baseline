#!/usr/bin/env bash
set -eu
: "${NDS_PROJECT_DIR:?set independent remote project directory}"
NDS_IMAGE="${NDS_IMAGE:-nds-vllm-source:kv-import-fix-20261012}"
NDS_CONTAINER=nds-mac17-v4flash-preflight
cd "$SHW_RESULTS_DIR"
npu-smi info > npu-before.txt
python3 - <<'PY'
import re
from pathlib import Path
idle=set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)\b',Path('npu-before.txt').read_text())))
assert set(range(8))<=idle,'Selected NPU occupied; abort before compute'
PY
docker image inspect "$NDS_IMAGE" --format '{{.Id}}' > image-id.txt
NDS_EXPECT_IMAGE_ID=sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270
test "$(cat image-id.txt)" = "$NDS_EXPECT_IMAGE_ID" || { echo 'Image ID differs from the validated runtime' >&2; exit 2; }
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-before.txt
for i in 0 1 2 3 4 5 6 7; do npu-smi info -t current-fault-event -i "$i" > "fault-device$i.txt" 2>&1 || true; done
cleanup() {
 docker rm -f "$NDS_CONTAINER" >/dev/null 2>&1 || true
 npu-smi info > npu-after.txt
 docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-after.txt
}
trap cleanup EXIT
set +e
timeout -k 20 600 docker run --rm --name "$NDS_CONTAINER" --privileged --network host \
 --security-opt label=disable --shm-size 4g --cpus 32 --memory 64g \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /etc/hccn.conf:/etc/hccn.conf:ro \
 -v "$SHW_RESULTS_DIR:/evidence" -v "$NDS_PROJECT_DIR/model:/model:ro" \
 -v "$NDS_PROJECT_DIR/scripts:/scripts:ro" \
 -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
 -e GLOO_SOCKET_IFNAME=lo \
 -e ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' -e OMP_NUM_THREADS=2 \
 -e VLLM_VERSION=0.23.0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 "$NDS_IMAGE" bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /usr/local/Ascend/nnal/atb/set_env.sh
 export PATH=/opt/nds-venv/bin:$PATH
 python - <<"PY"
import socket
with socket.socket() as s:s.bind(("127.0.0.1",18217))
PY
 timeout -k 10 180 python -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=18217 \
  /scripts/device_smoke.py > /evidence/smoke.log 2>&1
 timeout -k 10 180 python /scripts/version_probe.py > /evidence/version-probe.log 2>&1
 '
NDS_RC=$?
echo "$NDS_RC" > container-exit-code.txt
exit "$NDS_RC"
