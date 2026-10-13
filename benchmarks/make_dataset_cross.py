"""Freeze AISBench random synthetic samples at exactly 1024 chat input tokens."""
import hashlib, json, pathlib, random
import numpy as np
import torch
from transformers import AutoTokenizer
from ais_bench.benchmark.datasets.synthetic import SyntheticDataset

ROOT = pathlib.Path('/evidence')
MODEL = '/model'
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

def chat_tokens(text):
    ids = tokenizer.apply_chat_template([{'role':'user','content':text}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    assert isinstance(ids,list) and all(isinstance(token,int) for token in ids)
    return ids

def normalize(text):
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(chat_tokens(text[:middle])) <= 1024:
            low = middle
        else:
            high = middle - 1
    text = text[:low]
    while len(chat_tokens(text)) < 1024:
        text += ' a'
    assert len(chat_tokens(text)) == 1024
    return text

manifests = []
for name, seed in [('warm',90000), ('warm-destination',90001), ('A0',42000), ('A1',42001), ('A2',42002), ('C0',52000), ('C1',52001), ('C2',52002)]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator = object.__new__(SyntheticDataset)
    data = generator.load({'Type':'tokenid', 'RequestCount':100,
        'TokenIdConfig':{'RequestSize':1024,'PrefixLen':0}}, model_path=MODEL)
    records = [{'question':normalize(row['question']), 'answer':'dummy', 'max_tokens':10}
               for row in data]
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode()
    (ROOT / (name+'.jsonl')).write_bytes(payload)
    manifests.append({'dataset':name,'seed':seed,'requests':len(records),
        'sha256':hashlib.sha256(payload).hexdigest(),
        'input_tokens':[len(chat_tokens(row['question'])) for row in records],
        'unique_chat_hashes':len({hashlib.sha256(json.dumps(chat_tokens(row['question'])).encode()).hexdigest() for row in records}),
        'first_block_sha256':[hashlib.sha256(json.dumps(chat_tokens(row['question'])[:128]).encode()).hexdigest() for row in records]})
assert len({h for manifest in manifests for h in manifest['first_block_sha256']}) == 800
(ROOT/'dataset-manifest.json').write_text(json.dumps({'generator':'AISBench SyntheticDataset tokenid',
    'normalization':'Trim random text and append single-token ASCII padding to include the Qwen chat template in 1024 tokens',
    'output_tokens':10, 'datasets':manifests}, indent=2))
print('Frozen 8 datasets, each 100 unique requests of exactly 1024 chat input tokens')
