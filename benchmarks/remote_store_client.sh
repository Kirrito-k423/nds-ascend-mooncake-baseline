#!/usr/bin/env bash
set -eu
: "${NDS_SOURCE_HOST:?set source node address}"
: "${NDS_DESTINATION_HOST:?set destination node address}"
: "${NDS_PROJECT_DIR:?set the remote project directory}"
export NDS_PROBE_ROLE=client
export NDS_LOCAL_HOST="$NDS_SOURCE_HOST"
export NDS_LOCAL_HOST="$NDS_DESTINATION_HOST"
cd "$SHW_RESULTS_DIR"
cp "$NDS_PROJECT_DIR"/remote_store_probe.py .
cp "$NDS_PROJECT_DIR"/device_gate.py .
npu-smi info > before.txt
python3 device_gate.py before.txt 0
trap 'docker rm -f nds-mac3-remote-store-client >/dev/null 2>&1 || true; npu-smi info > after.txt' EXIT
timeout -k 10 180 docker run --rm --name nds-mac3-remote-store-client --privileged --network host --security-opt label=disable --shm-size 1g --cpus 4 --memory 8g \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /usr/local/dcmi:/usr/local/dcmi:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro -v /etc/hccn.conf:/etc/hccn.conf:ro \
 -v "$SHW_RESULTS_DIR:/evidence" -e NDS_PROBE_ROLE -e NDS_LOCAL_HOST -e NDS_SOURCE_HOST -e ASCEND_RT_VISIBLE_DEVICES=0 -e ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' \
 nds-vllm-source:mac3-20261012 bash -lc \
 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; /opt/nds-venv/bin/python /evidence/remote_store_probe.py "$NDS_PROBE_ROLE" "$NDS_LOCAL_HOST" "$NDS_SOURCE_HOST"'
