#!/usr/bin/env bash
set -eu
: "${NDS_SOURCE_HOST:?set source node address}"
: "${NDS_DESTINATION_HOST:?set destination node address}"
: "${NDS_PROJECT_DIR:?set the remote project directory}"
export NDS_INFERENCE_IMAGE="${NDS_INFERENCE_IMAGE:-nds-vllm-source:kv-import-fix-20261012}"
cd "$SHW_RESULTS_DIR"
cp "$NDS_PROJECT_DIR"/device_gate.py .
npu-smi info > before.txt
python3 device_gate.py before.txt 0,1,2,3
trap 'docker rm -f nds-mac3-qwen72b-cross-destination >/dev/null 2>&1 || true; npu-smi info > after.txt' EXIT
timeout -k 30 3600 python3 "$NDS_PROJECT_DIR"/run_cross_node.py destination
