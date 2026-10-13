"""Check only the completed experiment's services and selected NPU cards."""
import json
import pathlib
import socket
import subprocess
import sys

role = sys.argv[1]
output = pathlib.Path(sys.argv[2])
ports = [18172, 50272, 50273] if role == "source" else [18172, 18174]
container = "nds-mac3-qwen72b-cross-" + role
running = subprocess.check_output(
    ["docker", "ps", "--format", "{{.Names}}"], text=True
).splitlines()
assert container not in running, container
for port in ports:
    with socket.socket() as sock:
        sock.settimeout(2)
        assert sock.connect_ex(("127.0.0.1", port)) != 0, port
npu = subprocess.check_output(["npu-smi", "info"], text=True)
output.with_suffix(".npu.txt").write_text(npu)
subprocess.run(
    [sys.executable, str(pathlib.Path(__file__).with_name("device_gate.py")),
     str(output.with_suffix(".npu.txt")), "0,1,2,3"],
    cwd=output.parent, check=True,
)
report = {
    "role": role, "owned_container_absent": container,
    "owned_ports_closed": ports, "selected_cards_idle": [0, 1, 2, 3],
    "warning_allowed": True,
}
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
