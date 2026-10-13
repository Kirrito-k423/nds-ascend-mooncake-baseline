"""Query real Mooncake replica metadata for keys observed in KV loads."""
import gc,json,pathlib,sys
import torch,torch_npu
from store_replica_utils import describe

keys_file, output_file, source = sys.argv[1:4]
keys=json.loads(pathlib.Path(keys_file).read_text())
assert keys, 'No measured KV load keys to audit'
torch.npu.set_device(0)
from mooncake.store import MooncakeDistributedStore
store=MooncakeDistributedStore()
report={'keys_requested':len(keys),'source_host':source,'replicas':{},'all_on_source':False}
try:
    assert store.setup(source,'P2PHANDSHAKE',0,0,'ascend','',source+':50272')==0
    descriptors=store.batch_get_replica_desc(keys)
    for key in keys:
        rows=describe(descriptors.get(key,[]))
        report['replicas'][key]=rows
        assert rows and all(row['memory_replica'] and source in json.dumps(row['memory']) for row in rows), (key,rows)
    report['all_on_source']=True
finally:
    pathlib.Path(output_file).write_text(json.dumps(report,indent=2))
    del store
    gc.collect()
print(json.dumps({'keys_audited':len(keys),'all_memory_replicas_on_source':True}))
