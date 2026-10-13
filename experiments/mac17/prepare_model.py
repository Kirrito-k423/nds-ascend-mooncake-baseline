"""Fetch only pinned config/tokenizer files for the full-structure dummy run."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MODEL='deepseek-ai/DeepSeek-V4-Flash'
REVISION='60d8d70770c6776ff598c94bb586a859a38244f1'
FILES=['config.json','generation_config.json','tokenizer_config.json','tokenizer.json']

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--proxy',default='http://127.0.0.1:7890');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':a.proxy} if a.proxy else {}))
    manifest={'model_id':MODEL,'revision':REVISION,'tensor_payload_bytes_downloaded':0,'files':[]}
    for name in FILES:
        url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        with opener.open(url,timeout=45) as response:body=response.read(8*1024*1024+1)
        assert len(body)<=8*1024*1024
        json.loads(body);(a.out/name).write_bytes(body)
        manifest['files'].append({'path':name,'url':url,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
    cfg=json.loads((a.out/'config.json').read_text())
    assert cfg['num_hidden_layers']==43 and cfg['hidden_size']==4096
    assert cfg['n_routed_experts']==256 and cfg['num_nextn_predict_layers']==1
    (a.out/'metadata-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
