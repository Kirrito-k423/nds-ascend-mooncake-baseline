#!/usr/bin/env bash
set -eu
: "${NDS_PROJECT_DIR:?set independent remote project directory}"
NDS_IMAGE="${NDS_IMAGE:-nds-vllm-source:kv-import-fix-20261012}"
NDS_CONTAINER="${NDS_CONTAINER:-nds-mac17-v4flash-base}"
case "$NDS_CONTAINER" in nds-mac17-v4flash-*) ;; *) exit 2 ;; esac
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
cleanup() {
 touch "$SHW_RESULTS_DIR/sampler.stop"
 if test -n "${NDS_SAMPLER_PID:-}"; then wait "$NDS_SAMPLER_PID" || true; fi
 docker rm -f "$NDS_CONTAINER" >/dev/null 2>&1 || true
 npu-smi info > npu-after.txt
 docker ps --format '{{.Names}} {{.Image}} {{.Status}}' > containers-after.txt
}
trap cleanup EXIT
python3 "$NDS_PROJECT_DIR/scripts/host_sampler.py" --task-exit-marker "$SHW_RESULTS_DIR/sampler.stop" --out "$SHW_RESULTS_DIR/host-hbm.jsonl" > host-sampler.log 2>&1 &
NDS_SAMPLER_PID=$!
set +e
timeout -k 30 3600 docker run --rm --name "$NDS_CONTAINER" --privileged --network host \
 --security-opt label=disable --shm-size 16g --cpus 64 --memory 384g \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro \
 -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /etc/hccn.conf:/etc/hccn.conf:ro \
 -v "$SHW_RESULTS_DIR:/evidence" -v "$NDS_PROJECT_DIR/model:/model:ro" \
 -v "$NDS_PROJECT_DIR/scripts:/scripts:ro" \
 -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
 -e GLOO_SOCKET_IFNAME=lo \
 -e ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' -e OMP_NUM_THREADS=4 \
 -e NDS_SERVER_ARGS="${NDS_SERVER_ARGS:-}" \
 -e VLLM_VERSION=0.23.0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e HCCL_CONNECT_TIMEOUT=120 -e HCCL_EXEC_TIMEOUT=120 \
 -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True -e MAC17_EVIDENCE=/evidence \
 "$NDS_IMAGE" bash -lc '
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /usr/local/Ascend/nnal/atb/set_env.sh
 export PATH=/opt/nds-venv/bin:$PATH
 export PYTHONPATH=/scripts:${PYTHONPATH:-}
 timeout -k 20 3300 python /scripts/run_server.py ${NDS_SERVER_ARGS:-} > /evidence/inference.log 2>&1
 '
NDS_RC=$?
echo "$NDS_RC" > container-exit-code.txt
touch "$SHW_RESULTS_DIR/sampler.stop"
wait "$NDS_SAMPLER_PID"
exit "$NDS_RC"
