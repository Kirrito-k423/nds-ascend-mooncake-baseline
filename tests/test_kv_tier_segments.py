"""Keep SDK control names separate from data transport endpoints."""
import importlib.util
import json
from pathlib import Path
import unittest

path=Path(__file__).parents[1]/'benchmarks/kv_tier/segments.py'
spec=importlib.util.spec_from_file_location('segments',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def line(name,endpoint='SOURCE:16158',pool=268435456,contributes=True):
    return 'INFO NDS_TRACE '+json.dumps({'op':'store_ready','segment_name':name,
        'transport_endpoint':endpoint,'contributes':contributes,'pool_bytes':pool})

class SegmentTest(unittest.TestCase):
    def test_worker_controls_require_all_original_clients(self):
        rows=[]
        for rank in range(4):
            rows.append('NDS_TRACE '+json.dumps({'op':'store_ready','segment_name':f'SOURCE:{12000+rank}',
                        'contributes':True,'pool_bytes':1024,'worker_pid':rank+100,
                        'control_path':f'/evidence/disk/worker-clear-{rank+100}-{12000+rank}.sock'}))
        self.assertEqual(len(module.contributing_controls('\n'.join(rows),'SOURCE')),4)
        with self.assertRaises(AssertionError):module.contributing_controls('\n'.join(rows[:3]),'SOURCE')

    def test_control_hostname_not_transport_endpoint(self):
        # W1 SDK reported these distinct control/data ports.
        self.assertEqual(module.contributing_segments(line('SOURCE:12550'),'SOURCE',1),['SOURCE:12550'])

    def test_zero_contribution_scheduler_is_excluded(self):
        text=line('SOURCE:12550')+'\n'+line('SOURCE:14000',pool=0,contributes=False)
        self.assertEqual(module.contributing_segments(text,'SOURCE',1),['SOURCE:12550'])

    def test_wrong_node_or_missing_rank_is_rejected(self):
        with self.assertRaises(AssertionError):module.contributing_segments(line('DEST:12550'),'SOURCE',1)
        with self.assertRaises(AssertionError):module.contributing_segments(line('SOURCE:12550'),'SOURCE',4)

if __name__=='__main__':unittest.main()
