"""Plot accepted summaries only; no interpolation of skipped configurations."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1])
data=json.loads((root/'summary.json').read_text())
assert data['validated'], 'Refuse unvalidated measurement charts'
phases=data['phases']
outputs=[]
configs=sorted({(p['input_tokens'],p['concurrency']) for p in phases})
for length,c in configs:
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for column,mode in enumerate(['memory','disk']):
        rows=[r for r in phases if r['mode']==mode and r['input_tokens']==length and r['concurrency']==c]
        rows.sort(key=lambda r:(r['tier'],r['role']))
        if not rows:
            for ax in axes[:,column]:
                ax.axis('off');ax.text(.5,.5,mode+': no accepted measurements',ha='center',va='center')
            continue
        labels=[r['tier']+' '+r['role'] for r in rows]
        for ax,metric,title in [(axes[0,column],'ttft_ms','TTFT'),(axes[1,column],'tpot_ms','Request TPOT')]:
            med=[r[metric+'_p50'] for r in rows];p95=[r[metric+'_p95'] for r in rows]
            x=list(range(len(rows)))
            ax.bar(x,med,color='#327fa8',label='P50')
            ax.scatter(x,p95,color='#bf6f35',marker='x',label='Sample P95')
            ax.set_xticks(x,labels,rotation=20,ha='right')
            ax.set_ylabel('milliseconds');ax.set_title(mode+' / '+title)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.legend()
            for i,r in enumerate(rows):ax.annotate('n='+str(r['samples']),(i,max(med[i],p95[i])),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle(f'Qwen2.5-72B dummy TP4 — input {length}, C={c}, output 32\nP95 is descriptive; three rounds, no confidence interval')
    name=f'tiers-L{length}-C{c}.png';fig.savefig(root/name,dpi=160);plt.close(fig);outputs.append(name)
(root/'charts.json').write_text(json.dumps(outputs,indent=2))
print('\n'.join(outputs))
