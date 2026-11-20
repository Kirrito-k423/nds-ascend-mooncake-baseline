"""Foreground host coordinator for one fenced B02-1/2 atomic reservation.

Existing host SSH authenticates the peer; no credential enters a container.
Stops at the first failure and collects every started peer before returning.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
import traceback
from sdk_layout_probe import is_complete

PREFIX='MAC16_LAYOUT_RESULT '
SSH=['ssh','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8',
     '-oServerAliveInterval=10','-oServerAliveCountMax=3','root@177.5.22.1']
KEY_BYTES=10*1024**2


class Endpoint:
    def __init__(self,args,out,peer=False):
        self.peer=peer;self.name=f'mac16-layout-{args[4]}-{args[0]}';self.remote_out=out
        command=['bash',str(Path(__file__).with_name('run_sdk_layout.sh')),*map(str,args),str(out),
                 DESTINATION if peer else '177.5.21.1']
        if peer:command=SSH+[shlex.join(command)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,bufsize=1)
        self.queue=queue.Queue();self.log=Path(os.environ['SHW_RESULTS_DIR'])/(self.name+'.log')
        self.terminals=[]
        def reader():
            with self.log.open('w') as f:
                for line in self.process.stdout:
                    f.write(line);f.flush()
                    # SDK native logging can interleave before the marker.
                    pos=line.find(PREFIX)
                    if pos>=0:
                        try:value=json.loads(line[pos+len(PREFIX):])
                        except json.JSONDecodeError:continue
                        if value.get('kind')=='terminal':self.terminals.append(value)
                        self.queue.put(value)
            self.queue.put({'kind':'eof'})
        self.reader=threading.Thread(target=reader);self.reader.start()

    def receive(self,kind,op=None,seconds=210):
        deadline=time.monotonic()+seconds
        while True:
            item=self.queue.get(timeout=max(.01,deadline-time.monotonic()))
            if item['kind'] in ['error','eof']:raise RuntimeError(item)
            if item['kind']==kind and (op is None or item.get('op')==op):return item
            raise RuntimeError(('Unexpected protocol message',item))

    def command(self,op,**extra):
        self.process.stdin.write(json.dumps({'op':op,**extra})+'\n');self.process.stdin.flush()
        return self.receive('result',op)['result']

    def stop(self):
        if self.process.poll() is None:
            try:self.process.stdin.write('{"op":"stop"}\n');self.process.stdin.flush();self.process.stdin.close()
            except (BrokenPipeError,OSError):pass
        try:self.process.wait(timeout=200)
        except subprocess.TimeoutExpired:
            cleanup=['docker','rm','-f',self.name]
            if self.peer:cleanup=SSH+[shlex.join(cleanup)]
            # Exact task-owned container only. Its removal terminates its workers.
            subprocess.run(cleanup,timeout=45,check=True,capture_output=True)
            self.process.wait(timeout=60)
        self.reader.join(timeout=10)
        assert not self.reader.is_alive(),'Log reader still active'
        return {'host':DESTINATION if self.peer else '177.5.21.1',
                'coordinator_child_pid':self.process.pid,'exit_code':self.process.returncode,
                'container':self.name,'probe_terminals':self.terminals}


def disk_bytes():
    return {p.parent.name:int(p.read_text().split()[2])*512 for p in Path('/sys/block').glob('nvme*n1/stat')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--destination',choices=['177.5.22.1','177.5.23.1'],required=True)
    parser.add_argument('--resume-four-worker',action='store_true',help='Use the recorded TP4/single-worker gate evidence; run remaining cases only')
    parser.add_argument('--local-only',action='store_true',help='No peer allocation or peer launch')
    parser.add_argument('--compute-peer-only',action='store_true')
    parser.add_argument('--cross-only',action='store_true')
    args=parser.parse_args();out=Path(os.environ['SHW_RESULTS_DIR']);root=Path(__file__).parent
    global DESTINATION,SSH
    DESTINATION=args.destination;SSH[-1]='root@'+DESTINATION
    peer_root=root/(args.run+'-peer')
    # Fresh task-owned directory; an uncertain receipt is queried, never retried.
    if not args.local_only:subprocess.run(SSH+['mkdir '+shlex.quote(str(peer_root))],check=True,timeout=20)
    report={'run':args.run,'cases':[],'model_requests':0,'peer_root':str(peer_root),
            'source':'177.5.21.1','destination':DESTINATION,'terminals':[],'gates':[],'start_wall_ns':time.time_ns()}
    def save(): (out/'group-completion.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        for peer in ([True] if args.compute_peer_only else ([] if args.resume_four_worker else [False,True])):
            run=args.run+('-peer-gate' if peer else '-source-gate')
            gate_out=peer_root/run if peer else out/run
            command=['bash',str(root/'run_sdk_layout.sh'),'gate','4','1','memory',run,str(gate_out),
                     DESTINATION if peer else '177.5.21.1']
            if peer:command=SSH+[shlex.join(command)]
            with (out/(run+'.log')).open('w') as log:
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1060)
            report['gates'].append({'host':DESTINATION if peer else '177.5.21.1','exit_code':p.returncode,'run':run});save()
            if peer:
                subprocess.run(['scp','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-r',
                                'root@'+DESTINATION+':'+str(gate_out),str(out/'peer-gate')],
                               check=True,timeout=90,capture_output=True)
                gate_out=out/'peer-gate'
            assert p.returncode==0,('Compute/HCCL gate failed',run,p.returncode)
            rows=[json.loads((gate_out/f'tp4-rank{rank}.json').read_text()) for rank in range(4)]
            report['gates'][-1]['ranks']=rows;save()
            assert p.returncode==0 and all(row['world_size']==4 and row['integer_matmul_exact'] and row['all_reduce_correct'] for row in rows)
        matrix=[(1,1,'memory',False),(1,1,'disk',False),(1,8,'disk',False),
                (4,8,'disk',False),(4,8,'disk',True)]
        for index,(workers,keys,tier,cross) in enumerate(matrix):
            if args.resume_four_worker and index<3:continue
            if args.local_only and cross:continue
            if args.cross_only and not cross:continue
            run=f'{args.run}-s{index}';endpoints=[]
            case={'workers':workers,'keys_per_worker':keys,'buffers_per_key':160,
                  'key_bytes':KEY_BYTES,'tier':tier,'cross':cross,'rounds':[],'run':run}
            report['cases'].append(case);save()
            try:
                source=Endpoint(['writer',workers,keys,tier,run],out/run);endpoints.append(source)
                case['source_ready']=source.receive('ready');save()
                case['put']=source.command('put');save()
                if cross:
                    reader=Endpoint(['reader',workers,keys,tier,run],peer_root/run,peer=True);endpoints.append(reader)
                    case['reader_ready']=reader.receive('ready');save()
                else:reader=source
                for rn in range(3):
                    if tier=='disk':proof=source.command('prepare')
                    else:
                        proof=source.command('audit')
                        assert len(proof)==workers*keys and all(v and all(d['memory'] and not d['local_disk'] and is_complete(d['status']) for d in v) for v in proof.values())
                    before=disk_bytes();begin=time.perf_counter_ns()
                    result=reader.command('read',round=rn)
                    end=time.perf_counter_ns();after=disk_bytes()
                    disk_read=sum(after[k]-before[k] for k in before)
                    row={'round':rn,'proof':proof,'read':result,'source_disk_before':before,
                         'source_disk_after':after,'source_disk_read_bytes':disk_read,
                         'source_counter_begin_ns':begin,'source_counter_end_ns':end,
                         'clocks':'source host monotonic window; peer monotonic timestamps not aligned',
                         'counter_scope':'whole cohort incl control; overlapping rank windows are not summed'}
                    case['rounds'].append(row);save()
                    if tier=='disk':assert disk_read>=workers*keys*KEY_BYTES,('Physical disk witness absent',disk_read)
                case['functional_pass']=True;save()
            except BaseException:
                case['error']=traceback.format_exc();save();raise
            finally:
                for endpoint in reversed(endpoints):
                    terminal=endpoint.stop();report['terminals'].append(terminal);save()
                    if endpoint.peer:
                        subprocess.run(['scp','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-r',
                                        'root@'+DESTINATION+':'+str(endpoint.remote_out),str(out/'peer-results')],
                                       check=True,timeout=90,capture_output=True)
            assert all(t['exit_code']==0 for t in report['terminals']), 'Nonzero endpoint terminal'
        report['functional_pass']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        report['finish_wall_ns']=time.time_ns();save()
        # No source/peer child remains active when this controller exits.
        print(json.dumps({'run':args.run,'functional_pass':report.get('functional_pass',False),
                          'terminals':report['terminals']}),flush=True)


if __name__=='__main__':main()
