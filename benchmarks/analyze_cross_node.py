"""Validate real AISBench requests, KV tokens, and physical replica ownership."""
import csv,json,pathlib,re,statistics,sys

ROOT=pathlib.Path(sys.argv[1]);rows=[]
run=json.loads((ROOT/'cross-node-result.json').read_text())
assert run.get('sampling_finished') is True and not run.get('error'),run.get('error')
by_label={r['label']:r for r in run['phases']}
for repeat in range(3):
    for kind in ['cold-source','local-source','cold-destination','remote-destination']:
        label=kind+'-'+str(repeat)
        assert by_label[label]['exit_code']==0,by_label[label]
        folder=ROOT/'aisbench'/label
        csvs=list(folder.glob('**/performances/qwen72b-dummy-tp4/*.csv'))
        assert len(csvs)==1,(label,csvs)
        with csvs[0].open() as file:metrics={r['Performance Parameters']:r for r in csv.DictReader(file)}
        files=list(folder.glob('**/performances/qwen72b-dummy-tp4/*_details.jsonl'))
        assert len(files)==1,(label,files)
        details=[json.loads(line) for line in files[0].read_text().splitlines()]
        row={'phase':label,'server':by_label[label]['server'],'requests':len(details),
             'successful_requests':sum(bool(d['success']) for d in details),
             'input_token_counts':sorted({d['input_tokens'] for d in details}),
             'output_token_counts':sorted({d['output_tokens'] for d in details}),
             'ttft_mean_ms':float(metrics['TTFT']['Average'].split()[0]),
             'ttft_median_ms':float(metrics['TTFT']['Median'].split()[0]),'metrics':metrics,
             'common':json.loads(csvs[0].with_suffix('.json').read_text())}
        assert row['requests']==row['successful_requests']==100,row
        assert row['input_token_counts']==[1024] and row['output_token_counts']==[10],row
        log=(ROOT/(label+'-server.log')).read_text()
        found=re.findall(r'KV pool load spec created req=(\S+) vllm_cached=(\d+) kvpool_cached=(\d+)',log)
        hits={request:(int(local),int(external)) for request,local,external in found}
        row['hit_requests']=len(hits)
        row['local_cache_token_counts']=sorted({v[0] for v in hits.values()})
        row['external_cache_token_counts']=sorted({v[1] for v in hits.values()})
        if kind.startswith('cold'):assert not hits,row
        else:
            assert len(hits)==100 and all(v==(0,1023) for v in hits.values()),row
            audit=json.loads((ROOT/(label+'-replicas.json')).read_text())
            assert audit['all_on_source'] is True and audit['keys_requested']>0,audit
            for key,replicas in audit['replicas'].items():
                assert replicas and all(r['memory_replica'] and run['source_host'] in json.dumps(r['memory']) for r in replicas),(key,replicas)
            row['audited_keys_on_source']=audit['keys_requested']
        row['new_triton_compile_warnings']=len(re.findall(r'AutoBlockify disabled for kernel .token_bin_counts_and_mask_kernel',log))
        rows.append(row)
for node in ['source','destination']:
    warm=run[node+'-batch-shapes']
    assert [r['batch_size'] for r in warm]==list(range(1,26))
    assert sum(r['requests'] for r in warm)==325
groups={kind:[r for r in rows if r['phase'].startswith(kind+'-')] for kind in ['cold-source','local-source','cold-destination','remote-destination']}
means={kind:statistics.mean(r['ttft_mean_ms'] for r in group) for kind,group in groups.items()}
cvs={kind:statistics.stdev(r['ttft_mean_ms'] for r in group)/means[kind] for kind,group in groups.items()}
summary={'cross_node_measured':True,'all_request_and_cache_checks_passed':True,
 'formal_requests':sum(r['requests'] for r in rows),'formal_successful_requests':sum(r['successful_requests'] for r in rows),
 'source_machine':run.get('machine','source'),'destination_machine':'destination','destination_warning_cards_used':run['destination_devices'],
 'enforce_eager':run.get('enforce_eager',False),
 'ttft_means_ms':means,'round_cv':cvs,
 'local_speedup':means['cold-source']/means['local-source'],
 'remote_speedup_vs_source_cold':means['cold-source']/means['remote-destination'],
 'remote_speedup_vs_destination_cold':means['cold-destination']/means['remote-destination'],
 'phases':rows}
(ROOT/'validated-cross-node-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='phases'},indent=2))
