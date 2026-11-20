"""Check that the probe covers the actual layer/KV/block tensor layout."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('layout_probe',Path(__file__).parents[1]/'benchmarks/sdk_layout_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class LayoutTests(unittest.TestCase):
    def test_incomplete_descriptor_cannot_pass_complete_gate(self):
        self.assertTrue(probe.is_complete('ReplicaStatus.COMPLETE'))
        self.assertFalse(probe.is_complete('ReplicaStatus.INCOMPLETE'))
    def test_full_tensor_coverage_and_block_gather(self):
        for blocks in [1,8]:
            base=2*1024**2
            pointers=probe.pointer_lists(base,blocks)
            self.assertEqual(probe.KEY_BYTES,10*1024**2)
            # Independently index [layer, K/V, block, token, KV-head, head-dim].
            for block in range(blocks):
                expected=[]
                for layer in range(80):
                    for kv in range(2):
                        element_offset=((layer*2+kv)*blocks+block)*128*2*128
                        expected.append(base+element_offset*2)
                self.assertEqual(pointers[block],expected)
            flat=sorted(p for row in pointers for p in row)
            self.assertEqual(flat,list(range(base,base+blocks*probe.KEY_BYTES,65536)))


if __name__=='__main__':unittest.main()
