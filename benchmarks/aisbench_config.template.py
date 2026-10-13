from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.datasets.custom import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

models = [dict(attr='service', type=VLLMCustomAPIChatStream, abbr='qwen72b-dummy-tp4',
    path='/model', model='qwen', request_rate=0, retry=1, host_ip='127.0.0.1',
    host_port=18172, max_out_len=10, batch_size=25, trust_remote_code=False,
    generation_kwargs=dict(temperature=0, ignore_eos=True))]
datasets = [dict(abbr='__DATASET__', type=CustomDataset, path='/evidence/__DATASET__.jsonl',
    reader_cfg=dict(input_columns=['question','max_out_len'], output_column='answer'),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='{question}'),
        retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)))]
