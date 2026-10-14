"""Foreground source-host owner of a fixed two-machine model reservation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback
import urllib.request

SSH=['ssh','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8',
     '-oServerAliveInterval=15','-oServerAliveCountMax=4','root@177.5.22.1']
ASSETS=Path('/home/t00906153/nds-mac16-b02-20261014')
DESTINATION='177.5.22.1'


def role_command(role,output,kit):
    # Read only the task-owned launch.env, then override explicit physical roles.
    host='177.5.21.1' if role=='source' else DESTINATION
    lines=['set -eu',f'mkdir {shlex.quote(str(output))}',
           f'printf "%s\\n" "$$" > {shlex.quote(str(output/"wrapper.pid"))}',
           'set -a',f'source {ASSETS}/launch.env','set +a',
           f'export NDS_SOURCE_HOST=177.5.21.1 NDS_DESTINATION_HOST={DESTINATION}',
           f'export NDS_ROLE={role} NDS_LOCAL_HOST={host}',
           f'export NDS_TOOLKIT_SHA256={kit} SHW_RESULTS_DIR={shlex.quote(str(output))}',
           "export NDS_MODES_JSON='[\"disk\"]' NDS_CONFIGS_JSON='[[1024,1]]'",
           f'if docker container inspect nds-mac16-{role} >/dev/null 2>&1; then exit 73; fi',
           'terminal() { code=$?; trap - EXIT; printf \'{"pid":%s,"exit_code":%s,"finished_epoch":%s}\\n\' "$$" "$code" "$(date +%s)" > "$SHW_RESULTS_DIR/wrapper-terminal.json"; exit "$code"; }',
           'trap terminal EXIT',f'bash {ASSETS}/kv_tier/launch.sh']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--kit',required=True)
    parser.add_argument('--destination',choices=['177.5.22.1','177.5.23.1'],required=True)
    args=parser.parse_args();out=Path(os.environ['SHW_RESULTS_DIR']);peer_out=Path(__file__).parent/(args.run+'-peer-model')
    global DESTINATION
    DESTINATION=args.destination;SSH[-1]='root@'+DESTINATION
    report={'run':args.run,'source':'177.5.21.1','destination':DESTINATION,'kit':args.kit,
            'start_wall_ns':time.time_ns(),'terminals':[],'process_observations':[]}
    processes=[]
    def save(): (out/'group-completion.json').write_text(json.dumps(report,indent=2))
    def observe(role):
        command=['docker','top','nds-mac16-'+role,'-eo','pid,ppid,etime,args']
        if role=='destination':command=SSH+[shlex.join(command)]
        p=subprocess.run(command,capture_output=True,text=True,timeout=20)
        report['process_observations'].append({'role':role,'at_wall_ns':time.time_ns(),'exit_code':p.returncode,'processes':p.stdout})
        save()
    try:
        for role,target in [('destination',peer_out),('source',out/'source-model')]:
            command=['bash','-c',role_command(role,target,args.kit)]
            if role=='destination':command=SSH+[shlex.join(command)]
            log=(out/(role+'-controller.log')).open('w')
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            processes.append((role,p,log));report[role+'_child_pid']=p.pid;save()
        source=processes[1][1];last=0
        while source.poll() is None:
            if time.monotonic()-last>30:
                for role,_,_ in processes:observe(role)
                last=time.monotonic()
            time.sleep(2)
        report['source_exit_code']=source.returncode;save()
        if source.returncode!=0:raise RuntimeError('Source model owner exited '+str(source.returncode))
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        peer=next((p for role,p,_ in processes if role=='destination'),None)
        if peer is not None and peer.poll() is None:
            # The source normally already requested shutdown; query lifecycle before cleanup.
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                request=urllib.request.Request('http://'+DESTINATION+':18198/shutdown',data=b'{}',headers={'Content-Type':'application/json'})
                with opener.open(request,timeout=10) as response:report['peer_shutdown_reply']=response.read().decode()
            except Exception as error:report['peer_shutdown_error']=str(error)
        for role,p,log in processes:
            try:p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # No re-launch: remove only the experiment's exact model container.
                cleanup=['docker','rm','-f','nds-mac16-'+role]
                if role=='destination':cleanup=SSH+[shlex.join(cleanup)]
                stopped=subprocess.run(cleanup,capture_output=True,text=True,timeout=45)
                report[role+'_forced_container_cleanup']={'exit_code':stopped.returncode,'stderr':stopped.stderr}
                # Destination server also gets a bounded launch.sh timeout.
                p.wait(timeout=5500)
            log.close();report['terminals'].append({'role':role,'coordinator_child_pid':p.pid,'exit_code':p.returncode});save()
        if peer is not None:
            copied=subprocess.run(['scp','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-r',
                                   'root@'+DESTINATION+':'+str(peer_out),str(out/'peer-model')],
                                  capture_output=True,text=True,timeout=120)
            report['peer_archive_copy_exit']=copied.returncode;save();assert copied.returncode==0,copied.stderr
        for role in ['source','destination']:
            check=['docker','container','inspect','nds-mac16-'+role]
            if role=='destination':check=SSH+[shlex.join(check)]
            checked=subprocess.run(check,capture_output=True,text=True,timeout=30)
            report[role+'_container_absent']=checked.returncode!=0;save()
            assert checked.returncode==1 and 'No such' in checked.stderr,('Cannot confirm owned container absent',role,checked.returncode,checked.stderr)
        report['finish_wall_ns']=time.time_ns();save()
        print(json.dumps({'run':args.run,'terminals':report['terminals'],'error':report.get('error')}),flush=True)
    assert all(row['exit_code']==0 for row in report['terminals'])


if __name__=='__main__':main()
