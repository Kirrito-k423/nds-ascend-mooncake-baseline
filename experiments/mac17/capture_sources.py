"""Capture pinned public metadata and source; never download tensor payloads."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--proxy', default='http://127.0.0.1:7890')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({
    'http': a.proxy, 'https': a.proxy} if a.proxy else {}))
entries = []

def fetch(name, url, limit=12*1024*1024):
    with opener.open(urllib.request.Request(url, headers={'User-Agent': 'MAC17-metadata-probe'}), timeout=45) as r:
        body = r.read(limit+1)
        if len(body) > limit:
            raise ValueError('metadata limit exceeded')
    path = a.out/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {'path': name, 'url': url, 'bytes': len(body),
            'sha256': hashlib.sha256(body).hexdigest()}

identities = []
for model in ['deepseek-ai/DeepSeek-V4-Flash', 'deepseek-ai/DeepSeek-V4-Flash-0731',
              'Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp']:
    folder = model.split('/')[-1]
    try:
        entry = fetch(folder+'/hf-model.json', 'https://huggingface.co/api/models/'+model)
    except Exception as exc:
        identities.append({'model_id':model, 'error_type':type(exc).__name__,
                           'http_status':getattr(exc,'code',None), 'weight_payload_downloaded':False})
        continue
    entries.append(entry)
    meta = json.loads((a.out/entry['path']).read_text())
    sha = meta['sha']
    identities.append({'model_id': model, 'revision': sha, 'weight_payload_downloaded': False})
    filenames = {s['rfilename'] for s in meta['siblings']}
    selected = [f for f in filenames if f in {
        'config.json', 'quant_model_description.json', 'hf_quant_config.json',
        'quantization_config.json', 'model.safetensors.index.json',
        'quant_model_weights.safetensors.index.json', 'tokenizer_config.json'}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        entries.extend(pool.map(lambda f: fetch(folder+'/'+f,
            f'https://huggingface.co/{model}/resolve/{sha}/{f}'), sorted(selected)))

entry = fetch('ascend/head.json', 'https://api.github.com/repos/vllm-project/vllm-ascend/commits/main')
entries.append(entry)
sha = json.loads((a.out/entry['path']).read_text())['sha']
files = ['.github/vllm-main-verified.commit',
         'docs/source/tutorials/models/DeepSeek-V4-Flash.md',
         'docs/source/user_guide/supported_models.md',
         'docs/source/user_guide/feature_support_matrix.md']
for f in files:
    try:
        entries.append(fetch('ascend/'+f,
            f'https://raw.githubusercontent.com/vllm-project/vllm-ascend/{sha}/{f}'))
    except Exception as exc:
        entries.append({'path':f,'error_type':type(exc).__name__})
(a.out/'identities.json').write_text(json.dumps({'models':identities, 'ascend_source_revision':sha},indent=2))
(a.out/'manifest.json').write_text(json.dumps({'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                             'files':entries},indent=2))
print(json.dumps(identities,indent=2))
print('ascend_source_revision',sha)
