"""Foreground host device sampler; exits with this task's exit marker."""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task-exit-marker',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=3700)
    a=p.parse_args();start=time.monotonic();count=0
    with a.out.open('w') as f:
        while time.monotonic()-start<a.timeout:
            r=subprocess.run(['npu-smi','info'],text=True,capture_output=True,timeout=15)
            record={'time_ns':time.time_ns(),'monotonic_s':time.monotonic(),
                    'exit_code':r.returncode,'stdout':r.stdout}
            f.write(json.dumps(record)+'\n');f.flush();count+=1
            if count%6==0:
                print(json.dumps({'samples':count,'elapsed_s':round(time.monotonic()-start,1),
                    'hbm_mib':re.findall(r'(\d+)\s*/\s*98304',r.stdout)}),flush=True)
            if a.task_exit_marker.exists():break
            time.sleep(5)
    print(json.dumps({'samples':count,'task_exit_marker_present':a.task_exit_marker.exists()}),flush=True)

if __name__=='__main__':main()
