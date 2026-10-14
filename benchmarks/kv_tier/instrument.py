"""Create a reversible backend override; retain the validated SDK import order."""
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
assert 'from mooncake.store import ReplicateConfig' in source
assert source.index('from mooncake.store import ReplicateConfig') > source.index('def put(')
source = source.replace('import threading\n', 'import threading\nimport time\nimport sys\nsys.path.insert(0,"/task")\nfrom worker_control import WorkerControl\n')
start = '        return store'
replacement = '''        nds_control_path = None
        self._nds_worker_control = None
        if self._contribute_memory and self.config.global_segment_size > 0:
            self._nds_worker_control = WorkerControl(store, os.environ["NDS_LOCAL_HOST"])
            nds_control_path = self._nds_worker_control.start("/evidence/disk") if os.environ.get("NDS_WORKER_CLEAR") == "1" else None
        logger.info("NDS_TRACE %s", json.dumps({"op": "store_ready",
                    "segment_name": store.get_hostname(), "transport_endpoint": self.local_seg,
                    "worker_pid": os.getpid(), "control_path": nds_control_path,
                    "contributes": self._contribute_memory,
                    "pool_bytes": self.config.global_segment_size if self._contribute_memory else 0}))
        return store'''
assert source.count(start) == 1
source = source.replace(start,replacement)
start = '            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)'
replacement = '''            nds_begin = time.perf_counter_ns()
            nds_rpc_before = self.store.get_offload_rpc_read_count()
            logger.info("NDS_TRACE %s", json.dumps({"op": "get_begin", "keys": keys, "sizes": sizes,
                        "begin_ns": nds_begin, "local_seg": self.local_seg}))
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
            logger.info("NDS_TRACE %s", json.dumps({"op": "get_end", "keys": keys,
                        "begin_ns": nds_begin, "end_ns": time.perf_counter_ns(),
                        "returns": list(res), "offload_rpc_before": nds_rpc_before,
                        "offload_rpc_after": self.store.get_offload_rpc_read_count(),
                        "local_seg": self.local_seg}))'''
assert source.count(start) == 1
source = source.replace(start, replacement)
start = '            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)'
replacement = '''            logger.info("NDS_TRACE %s", json.dumps({"op": "put", "keys": keys, "sizes": sizes,
                        "begin_ns": time.perf_counter_ns(), "local_seg": self.local_seg}))
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            if self._nds_worker_control is not None:
                self._nds_worker_control.remember(keys, list(res))'''
assert source.count(start) == 1
Path(sys.argv[2]).write_text(source.replace(start, replacement))
