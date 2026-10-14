"""Reject plausible false tier labels. Fixtures are synthetic unit-test inputs."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('analyze',Path(__file__).parents[1]/'benchmarks/kv_tier/analyze.py')
analyze=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyze)


class EvidenceTest(unittest.TestCase):
    def fixture(self):
        replicas={'key':[{'status':'COMPLETE','memory':False,'local_disk':True}]}
        evidence={'tier':'T3','input_tokens':1024,'requests':1,'loads':[['req',0,1023]],
                  'role':'source',
                  'before_replicas':replicas,'traces':[{'op':'get_end','keys':['key'],'returns':[1048576],
                    'offload_rpc_before':0,'offload_rpc_after':1}],
                  'before':{'source':{'disk':{'nvme0n1':{'read_bytes':0}},'metrics':'vllm:request_prefill_kv_computed_tokens_sum 0\nvllm:prefix_cache_hits_total 0'}},
                  'after':{'source':{'disk':{'nvme0n1':{'read_bytes':1048576}},'metrics':'vllm:request_prefill_kv_computed_tokens_sum 1\nvllm:prefix_cache_hits_total 0'}}}
        requests=[{'success':True,'usage':{'prompt_tokens':1024,'completion_tokens':32},'token_timing_exact':True}]
        return evidence,requests

    def test_disk_only_witness(self):
        e,r=self.fixture();self.assertEqual(analyze.validate(e,r,'SOURCE'),1048576)

    def test_dram_refill_cannot_be_ssd(self):
        e,r=self.fixture();e['before_replicas']['key'][0]['memory']=True
        with self.assertRaises(AssertionError):analyze.validate(e,r,'SOURCE')

    def test_no_disk_read_rejected(self):
        e,r=self.fixture();e['after']['source']['disk']['nvme0n1']['read_bytes']=0
        with self.assertRaises(AssertionError):analyze.validate(e,r,'SOURCE')

    def test_partial_load_rejected(self):
        e,r=self.fixture();e['loads'][0][2]=896
        with self.assertRaises(AssertionError):analyze.validate(e,r,'SOURCE')

    def test_unaudited_key_rejected(self):
        e,r=self.fixture();e['traces'][0]['keys']=['other-key']
        with self.assertRaises(AssertionError):analyze.validate(e,r,'SOURCE')

    def test_failed_request_rejected(self):
        e,r=self.fixture();r[0]['success']=False
        with self.assertRaises(AssertionError):analyze.validate(e,r,'SOURCE')

    def test_incomplete_status_cannot_match_complete(self):
        e,r=self.fixture();e['before_replicas']['key'][0]['status']='ReplicaStatus.INCOMPLETE'
        with self.assertRaises(AssertionError):analyze.validate(e,r,'SOURCE')


if __name__=='__main__':unittest.main()
