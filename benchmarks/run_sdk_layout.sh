#!/usr/bin/env bash
# Task-owned wrapper. The host coordinator supplies an absolute fresh result directory.
set -eu
role=$1; workers=$2; keys=$3; tier=$4; run_id=$5; output=$6; host=$7
helper_root=$(cd "$(dirname "$0")" && pwd)
asset_root=/home/t00906153/nds-mac16-b02-20261014
name="mac16-layout-$run_id-$role"
mkdir "$output"
cd "$output"
printf '%s\n' "$$" > "$output/wrapper.pid"
if docker container inspect "$name" >/dev/null 2>&1; then exit 73; fi
cleanup() {
  code=$?
  trap - EXIT
  docker rm -f "$name" > "$output/container-cleanup.txt" 2>&1 || true
  npu-smi info > "$output/after.txt" || true
  printf '{"pid":%s,"exit_code":%s,"container":"%s","finished_epoch":%s}\n' "$$" "$code" "$name" "$(date +%s)" > "$output/wrapper-terminal.json"
  # Retain descriptors, file sizes, seeds, SHA and logs; remove only this invocation's disposable payload.
  if [ -d "$output/offload" ]; then rm -rf -- "$output/offload"; fi
  exit "$code"
}
trap cleanup EXIT
python3 - "$asset_root" <<'PY'
import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path(sys.argv[1]);m=json.loads((r/'staging-manifest.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','nds-vllm-source:kv-import-fix-20261012'],text=True).strip()=='sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270'
PY
npu-smi info > "$output/before.txt"
python3 "$asset_root/kv_tier/idle_gate.py" "$output/before.txt" 0,1,2,3
findmnt -J -T "$output" > "$output/mount.json"
if [ "$role" = writer ]; then
  python3 - <<'PY'
import socket
for port in [50326,50327]:
 with socket.socket() as s:s.bind(('0.0.0.0',port))
PY
fi
timeout -k 15 1000 docker run --rm -i --name "$name" --privileged --network host --security-opt label=disable --shm-size 2g --cpus 16 --memory 64g \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "$output:/evidence" -v "$helper_root:/layout:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 -e 'ASCEND_LOCAL_COMM_RES={"version":"1.3"}' \
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=134217728 -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=536870912 \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=536870912 -e MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES=268435456 \
  -e MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1 -e MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=1 -e MOONCAKE_OFFLOAD_USE_URING=false \
  -e NDS_LAYOUT_ROLE="$role" -e NDS_LAYOUT_WORKERS="$workers" -e NDS_LAYOUT_KEYS="$keys" \
  -e NDS_LAYOUT_TIER="$tier" -e NDS_LAYOUT_RUN="$run_id" -e NDS_LAYOUT_HOST="$host" \
  nds-vllm-source:kv-import-fix-20261012 bash -lc '
set -e
ulimit -c 0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/opt/nds-venv/bin:$PATH
if [ "$NDS_LAYOUT_ROLE" = writer ]; then
  mooncake_master --port=50326 --metrics_port=50327 --enable_offload=true --default_kv_lease_ttl=5000 > /evidence/master.log 2>&1 &
  master_pid=$!
  trap "kill $master_pid 2>/dev/null || true; wait $master_pid 2>/dev/null || true" EXIT
  sleep 2
  kill -0 "$master_pid"
fi
if [ "$NDS_LAYOUT_ROLE" = gate ]; then
  /opt/nds-venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29596 /layout/tp4_gate.py
else
  python3 /layout/sdk_layout_probe.py --role "$NDS_LAYOUT_ROLE" --workers "$NDS_LAYOUT_WORKERS" --keys "$NDS_LAYOUT_KEYS" --tier "$NDS_LAYOUT_TIER" --run "$NDS_LAYOUT_RUN" --out /evidence --host "$NDS_LAYOUT_HOST" --master 177.5.21.1:50326 --clear-via-control
fi
'
