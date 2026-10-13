"""Regression for native SDK initialization before an NPU context exists."""

import subprocess
import sys


def test_import_backend_before_device_initialization():
    code = (
        "from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend "
        "import MooncakeBackend; print('backend-import-complete')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "backend-import-complete" in result.stdout
