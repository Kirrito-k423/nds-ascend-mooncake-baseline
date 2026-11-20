"""Reproduce Master's client-ownership rule with four independent publishers."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).parents[1]/'benchmarks/kv_tier/worker_control.py'
spec=importlib.util.spec_from_file_location('worker_control',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Master:
    def __init__(self):self.memory={f'key{rank}':rank for rank in range(4)}


class Client:
    def __init__(self,master,rank):self.master=master;self.rank=rank
    def get_hostname(self):return f'SOURCE:{12000+self.rank}'
    def batch_replica_clear(self,keys,segment):
        result=[]
        for key in keys:
            owner=self.master.memory.get(key)
            if owner is not None and owner!=self.rank:result.append(-1)
            else:
                if owner==self.rank and segment==self.get_hostname():self.master.memory.pop(key)
                result.append(0)
        return result


class WorkerControlTests(unittest.TestCase):
    def test_correct_segment_names_do_not_authorize_another_client(self):
        master=Master();clients=[Client(master,r) for r in range(4)]
        for client in clients:clients[0].batch_replica_clear(list(master.memory),client.get_hostname())
        self.assertEqual(set(master.memory),{'key1','key2','key3'})

    def test_each_successful_publisher_clears_its_own_keys(self):
        master=Master();clients=[Client(master,r) for r in range(4)]
        controls=[module.WorkerControl(c,'SOURCE') for c in clients]
        for rank,control in enumerate(controls):control.remember([f'key{rank}'],[0])
        covered=set()
        for control in controls:
            result=control.clear([f'key{r}' for r in range(4)],[c.get_hostname() for c in clients])
            covered.update(result['owned_keys'])
        self.assertEqual(covered,{f'key{r}' for r in range(4)})
        self.assertEqual(master.memory,{})

    def test_failed_publications_and_foreign_segments_are_rejected(self):
        c=module.WorkerControl(Client(Master(),0),'SOURCE');c.remember(['key0'],[-1])
        self.assertEqual(c.clear(['key0'],['SOURCE:12000'])['owned_keys'],[])
        with self.assertRaises(AssertionError):c.clear(['key0'],['OTHER:12000'])


if __name__=='__main__':unittest.main()
