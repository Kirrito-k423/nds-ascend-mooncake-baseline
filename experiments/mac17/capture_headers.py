"""Read only indexed safetensors headers, requiring HTTP 206 ranges."""
import argparse
import concurrent.futures
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--model-id',default='deepseek-ai/DeepSeek-V4-Flash')
p.add_argument('--revision',required=True)
p.add_argument('--index',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--proxy',default='http://127.0.0.1:7890')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
index=json.loads(a.index.read_text());names=sorted(set(index['weight_map'].values()))
op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':a.proxy} if a.proxy else {}))

def read_range(url,first,last):
    req=urllib.request.Request(url,headers={'Range':f'bytes={first}-{last}'})
    with op.open(req,timeout=45) as r:
        if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {first}-{last}/'):
            raise ValueError('server did not honor metadata-only range')
        b=r.read(last-first+2)
        if len(b)!=last-first+1:raise ValueError('incomplete/oversized range')
        return b

def capture(name):
    url=f'https://huggingface.co/{a.model_id}/resolve/{a.revision}/{name}'
    result={'shard':name,'url':url,'tensor_payload_bytes_read':0}
    try:
        size=struct.unpack('<Q',read_range(url,0,7))[0]
        if not 0<size<=4*1024*1024:raise ValueError('invalid header length')
        body=read_range(url,8,7+size);header=json.loads(body)
        (a.out/(name+'.header.json')).write_bytes(body)
        result.update(header_bytes=size,sha256=hashlib.sha256(body).hexdigest(),
            tensor_count=sum(k!='__metadata__' for k in header),
            source_payload_bytes=sum(v['data_offsets'][1]-v['data_offsets'][0]
                                     for k,v in header.items() if k!='__metadata__'))
    except Exception as exc:
        result.update(error_type=type(exc).__name__,http_status=getattr(exc,'code',None))
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    records=list(pool.map(capture,names))
(a.out/'manifest.json').write_text(json.dumps(records,indent=2))
summary={'model_id':a.model_id,'revision':a.revision,'indexed_shards':len(names),
         'captured_shards':sum('error_type' not in r for r in records),
         'tensor_count':sum(r.get('tensor_count',0) for r in records),
         'source_payload_bytes':sum(r.get('source_payload_bytes',0) for r in records),
         'index_total_size':index['metadata']['total_size'],'tensor_payload_bytes_read':0}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if summary['captured_shards']!=len(names):raise SystemExit(1)
