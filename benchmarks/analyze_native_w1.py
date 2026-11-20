"""Validate every recorded W1 native sample and plot samples plus min/max."""
import argparse
import csv
import json
from pathlib import Path
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('data',type=Path)
args=parser.parse_args()
rows=[]
for name in ['cold-memory','cold-disk','boundary-memory','boundary-disk','warm-disk']:
    value=json.loads((args.data/(name+'.json')).read_text())
    assert value['setup_return']==0
    assert value.get('functional_pass') or name=='warm-disk'
    for r in value['records']:
        size=r['size_bytes']
        assert r['exact'] and r['sha256']==r['expected_sha256']
        assert r['returned_bytes']==size
        if value['mode']=='disk':
            proof=r['replicas']
            assert proof['lease_wait_s']==6
            assert all(x['local_disk'] and not x['memory'] and
                       x['status'].rsplit('.',1)[-1]=='COMPLETE' for x in proof['after'])
            assert r['offload_rpc_delta']>0
            if name!='warm-disk':
                assert proof['own_files_fsync_fadvise']
                assert not proof['global_page_cache_modified']
                assert r['disk_read_bytes']>=size
        rows.append({'run':name,'mode':value['mode'],'size_bytes':size,'round':r['round'],
                     'elapsed_ms':r['elapsed_ms'],'disk_read_bytes':r['disk_read_bytes'],
                     'rpc_reads':r['offload_rpc_delta'],'exact':r['exact'],
                     'sha256':r['sha256'],'physical_read':r['disk_read_bytes']>=size})
with (args.data/'native-reads.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
groups={}
for r in rows:groups.setdefault((r['run'],r['size_bytes']),[]).append(r)
summary=[]
for (run,size),values in groups.items():
    assert len(values)==3 and sorted(v['round'] for v in values)==[0,1,2]
    samples=[v['elapsed_ms'] for v in values]
    baseline=groups.get((run.replace('disk','memory'),size))
    mean=statistics.mean(samples)
    summary.append({'run':run,'size_bytes':size,'n':len(values),'samples_ms':samples,
                    'mean_ms':mean,'median_ms':statistics.median(samples),
                    'min_ms':min(samples),'max_ms':max(samples),
                    'change_from_matched_memory_pct':None if not baseline else
                     (mean/statistics.mean(v['elapsed_ms'] for v in baseline)-1)*100,
                    'disk_read_bytes_each':[v['disk_read_bytes'] for v in values],
                    'rpc_reads_each':[v['rpc_reads'] for v in values]})
(args.data/'native-summary.json').write_text(json.dumps({'groups':summary,
    'total_completed_reads':len(rows),'accepted_physical_ssd_reads':sum(
       r['mode']=='disk' and r['physical_read'] for r in rows),
    'model_requests':0,'failed_64MiB_read_included_as_performance_sample':False},indent=2))
print(json.dumps({'total_completed_reads':len(rows),'groups':summary},indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax in axes:
    ax.set_xlabel('Native object size (MiB)');ax.grid(alpha=.2)
for mode,color in [('memory','#2058a5'),('disk','#c55222')]:
    selected=sorted((g for g in summary if g['run'] in ['cold-'+mode,'boundary-'+mode]),
                    key=lambda g:g['size_bytes'])
    x=[g['size_bytes']/1024**2 for g in selected]
    y=[g['mean_ms'] for g in selected]
    axes[0].errorbar(x,y,yerr=[[g['mean_ms']-g['min_ms'] for g in selected],
                              [g['max_ms']-g['mean_ms'] for g in selected]],
                     fmt='o',capsize=5,color=color,label=mode.upper()+' mean / range')
    for g in selected:
        axes[0].scatter([g['size_bytes']/1024**2]*3,g['samples_ms'],s=20,
                        marker='x',color=color,alpha=.6)
    axes[1].scatter(x,[statistics.mean(g['disk_read_bytes_each'])/1024**2 for g in selected],
                    color=color,label=mode.upper()+' measured')
axes[1].plot([1,3,10],[1,3,10],'--',color='grey',label='Payload reference')
axes[0].set_ylabel('get_into + NPU synchronize (ms)')
axes[1].set_ylabel('Host NVMe read delta per call (MiB)')
axes[0].set_title('All n=3 samples; initialization excluded')
axes[1].set_title('SSD: each payload + 4096 B; shared host counter')
for ax in axes:ax.set_xticks([1,3,10]);ax.legend(fontsize=8)
fig.suptitle('MAC-16 W1 native readback: single NPU, same image; no model requests')
fig.tight_layout()
fig.savefig(args.data.parent/'native-w1.png',dpi=170)
