"""Validate fio evidence and summarize measured bytes and worker latency ranges."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(directory):
    allocation = json.loads((directory / 'file-allocation.json').read_text())
    assert allocation['size'] == 32 * 1024**3
    assert allocation['allocated_bytes'] >= allocation['size']
    assert json.loads((directory / 'cleanup.json').read_text())['test_file_removed']
    rows = []
    for file in sorted(directory.glob('*-r*.json')):
        if '.samples.' in file.name or '.exit.' in file.name:
            continue
        name = file.stem
        record = json.loads(file.read_text())
        jobs = record['jobs']
        assert all(job['error'] == 0 for job in jobs), name
        assert json.loads((directory / (name + '.exit.json')).read_text())['exit_code'] == 0
        options = [dict(record.get('global options', {}), **job.get('job options', {})) for job in jobs]
        assert all(option['direct'] == '1' for option in options)
        assert all(option['ioengine'] == 'io_uring' for option in options)
        assert all(job['write']['io_bytes'] == 0 for job in jobs)
        samples = json.loads((directory / (name + '.samples.json')).read_text())
        before, after = samples[0], samples[-1]
        reads = {disk: 512 * (counters[2] - before['diskstats'][disk][2])
                 for disk, counters in after['diskstats'].items() if 'p' not in disk}
        writes = {disk: 512 * (counters[6] - before['diskstats'][disk][6])
                  for disk, counters in after['diskstats'].items() if 'p' not in disk}
        measured_bytes = sum(job['read']['io_bytes'] for job in jobs)
        assert reads['nvme0n1'] >= 0.98 * measured_bytes, (name, reads, measured_bytes)
        cpu_before = list(map(int, before['cpu'].split()[1:9]))
        cpu_after = list(map(int, after['cpu'].split()[1:9]))
        cpu = [a - b for a, b in zip(cpu_after, cpu_before)]
        latencies = [job['read']['lat_ns']['percentile'] for job in jobs]
        row = {
            'name': name, 'workers': len(jobs),
            'bandwidth_Bps': sum(job['read']['bw_bytes'] for job in jobs),
            'iops': sum(job['read']['iops'] for job in jobs),
            'measured_read_bytes': measured_bytes, 'physical_read_bytes': reads,
            'physical_write_bytes': writes,
            'fio_cpu_percent_sum': sum(job['usr_cpu'] + job['sys_cpu'] for job in jobs),
            'host_cpu_busy_percent': 100 * (sum(cpu) - cpu[3] - cpu[4]) / sum(cpu),
            'host_iowait_percent': 100 * cpu[4] / sum(cpu),
            'sample_duration_s': after['monotonic'] - before['monotonic'],
            'worker_latency_percentiles_ns': latencies,
            'iodepth_level': [job['iodepth_level'] for job in jobs],
        }
        for percentile in (50, 95, 99):
            values = [latency[f'{percentile:.6f}'] for latency in latencies]
            row[f'p{percentile}_worker_min_us'] = min(values) / 1000
            row[f'p{percentile}_worker_max_us'] = max(values) / 1000
        rows.append(row)
    assert len(rows) == 12, len(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row['name'].rsplit('-r', 1)[0], []).append(row)
    summary = {}
    for name, group in groups.items():
        assert len(group) == 3
        bandwidth = [row['bandwidth_Bps'] / 10**9 for row in group]
        iops = [row['iops'] for row in group]
        summary[name] = {'bandwidth_GBs': bandwidth, 'mean_GBs': statistics.mean(bandwidth),
            'cv_percent': 100 * statistics.stdev(bandwidth) / statistics.mean(bandwidth),
            'mean_iops': statistics.mean(iops),
            'p50_worker_range_us': [min(r['p50_worker_min_us'] for r in group), max(r['p50_worker_max_us'] for r in group)],
            'p95_worker_range_us': [min(r['p95_worker_min_us'] for r in group), max(r['p95_worker_max_us'] for r in group)],
            'p99_worker_range_us': [min(r['p99_worker_min_us'] for r in group), max(r['p99_worker_max_us'] for r in group)],
            'mean_fio_cpu_percent_sum': statistics.mean(r['fio_cpu_percent_sum'] for r in group)}
    return {'validated': True, 'rows': rows, 'summary': summary,
            'latency_note': 'Four-worker percentiles are reported per worker as a range; they are not pooled percentiles.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))
