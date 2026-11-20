"""Rebuild complete native samples from already extracted private task archives."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-root',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();base=Path(args.raw_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];statuses=[]
    for run in ['c1','c2','c3','c4']:
        report=json.loads((base/('layout-'+run)/'results/group-completion.json').read_text())
        terminal=json.loads((base/('task-'+run+'-terminal.json')).read_text())
        statuses.append({'run':run,'task_id':terminal['id'],'exit_code':terminal['exitCode'],'archiveReady':terminal['archiveReady'],
                         'cases':[{'run':c['run'],'tier':c['tier'],'workers':c['workers'],'keys_per_worker':c['keys_per_worker'],
                                   'cross':c['cross'],'cohorts':len(c['rounds']),'functional_samples_pass':c.get('functional_pass',False),
                                   'error':c.get('error')} for c in report['cases']],
                         'error':report.get('error'),'terminals':report['terminals'],'gates':report['gates']})
        for case in report['cases']:
            for rnd in case['rounds']:
                workers=rnd['read']['workers']
                span=(max(w['end_ns'] for w in workers)-min(w['begin_ns'] for w in workers))/1e6
                for w in workers:
                    rows.append({'run':run,'case':case['run'],'task_exit':terminal['exitCode'],'tier':case['tier'],'cross':case['cross'],
                                 'workers':case['workers'],'keys_per_worker':case['keys_per_worker'],'key_bytes':case['key_bytes'],
                                 'round':rnd['round'],'rank':w['rank'],'native_ms':w['elapsed_ms'],'native_span_ms':span,
                                 'rpc_delta':w['rpc_delta'],'returned_bytes':sum(w['returns']),'key_reads':len(w['keys']),
                                 'all_exact_sha':all(k['exact'] and k['sha256']==k['expected_sha256'] for k in w['keys']),
                                 'source_disk_read_bytes_cohort':rnd['source_disk_read_bytes'],
                                 'source_counter_ms':(rnd['source_counter_end_ns']-rnd['source_counter_begin_ns'])/1e6})
    with (out/'native-layout-samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/'native-layout-status.json').write_text(json.dumps(statuses,indent=2))
    groups=[]
    for case in dict.fromkeys(row['case'] for row in rows):
        rr=[r for r in rows if r['case']==case];first=rr[0];cohorts=[r for r in rr if r['rank']==0]
        group={k:first[k] for k in ['run','case','task_exit','tier','cross','workers','keys_per_worker','key_bytes']}
        group.update({'cohorts':len(cohorts),'native_calls':len(rr),'key_reads':sum(r['key_reads'] for r in rr),
                      'native_span_mean_ms':statistics.mean(r['native_span_ms'] for r in cohorts),
                      'native_span_samples_ms':[r['native_span_ms'] for r in cohorts],
                      'rank_native_mean_ms':statistics.mean(r['native_ms'] for r in rr),
                      'disk_samples_bytes':[r['source_disk_read_bytes_cohort'] for r in cohorts],
                      'all_exact_sha':all(r['all_exact_sha'] for r in rr),'rpc_delta_total':sum(r['rpc_delta'] for r in rr)})
        groups.append(group)
    (out/'native-layout-summary.json').write_text(json.dumps({'model_requests':0,'direct_dma_tested':False,
              'counter_scope':'whole source host per cohort; repeated across rank rows, never sum duplicate windows','groups':groups},indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    labels=['MEM 1 worker / 1 key','SSD 1 worker / 1 key','SSD 1 worker / 8 keys',
            'SSD 4 workers / 8 keys\nC3: cleanup failure','SSD 4 workers / 8 keys\nC4: local, exit 0',
            'SSD 4 workers / 8 keys\nC4: remote, exit 0']
    figure,axis=plt.subplots(figsize=(11,5))
    for x,group in enumerate(groups):
        values=group['native_span_samples_ms'];axis.scatter([x-.09,x,x+.09],values,s=40,color='#3568a8',label='Individual rounds' if x==0 else None)
        axis.plot([x-.2,x+.2],[group['native_span_mean_ms']]*2,color='#d66b35',linewidth=2,label='Mean' if x==0 else None)
    axis.set_xticks(range(len(labels)),labels,fontsize=8)
    axis.set_ylabel('Native call + device sync cohort span (ms)')
    axis.set_title('10MiB/key, 160 buffers/key; all 3 samples retained; no model TTFT')
    axis.grid(axis='y',alpha=.25);axis.legend();figure.tight_layout();figure.savefig(out/'native-layout-latency.png',dpi=160);plt.close(figure)


if __name__=='__main__':main()
