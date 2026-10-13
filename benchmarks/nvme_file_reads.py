"""Bounded file-only fio reads. Run only after reserving an idle disk window."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def snapshot():
    disks = {}
    for line in Path('/proc/diskstats').read_text().splitlines():
        fields = line.split()
        if fields[2].startswith('nvme'):
            disks[fields[2]] = list(map(int, fields[3:]))
    return {'time': time.time(), 'monotonic': time.monotonic(), 'diskstats': disks,
            'cpu': Path('/proc/stat').read_text().splitlines()[0]}


def run_fio(fio, config, out, name, readonly=True):
    job = out / (name + '.fio')
    job.write_text(config)
    command = [str(fio), '--output-format=json', '--output=' + str(out / (name + '.json'))]
    if readonly:
        command.append('--readonly')
    command.append(str(job))
    samples = [snapshot()]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.5):
            samples.append(snapshot())

    worker = threading.Thread(target=sample)
    worker.start()
    try:
        with (out / (name + '.stderr')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=log, timeout=180)
    finally:
        stop.set()
        worker.join()
        samples.append(snapshot())
        (out / (name + '.samples.json')).write_text(json.dumps(samples, indent=2))
    (out / (name + '.exit.json')).write_text(json.dumps({'command': command, 'exit_code': result.returncode}))
    if result.returncode:
        raise RuntimeError('fio failed: ' + name)
    data = json.loads((out / (name + '.json')).read_text())
    assert data['jobs'] and all(job['error'] == 0 for job in data['jobs'])
    print(name, 'complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fio', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpus', required=True, help='Reserved CPU set local to the selected NVMe NUMA node')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.directory.mkdir(parents=True, exist_ok=True)
    file = args.directory.resolve() / 'nvme-read-only-test.bin'
    size = 32 * 1024**3
    assert not file.exists(), 'Refuse to overwrite any existing file'
    stat = os.statvfs(args.directory)
    assert stat.f_bavail * stat.f_frsize > size + 100 * 1024**3
    mapping = subprocess.check_output(['findmnt', '-J', '-T', str(args.directory)], text=True)
    (args.output / 'file-mount.json').write_text(mapping)
    mount = json.loads(mapping)['filesystems'][0]
    assert mount['fstype'] not in ('tmpfs', 'overlay', 'nfs', 'nfs4'), mount
    common = ('[global]\nioengine=io_uring\ndirect=1\nthread=1\niodepth=32\n'
              'invalidate=0\nnumjobs=1\nsize=32G\nfilename=' + str(file) + '\n'
              'cpus_allowed=' + args.cpus + '\ncpus_allowed_policy=split\n'
              'percentile_list=50:95:99\nlat_percentiles=1\n')
    try:
        run_fio(args.fio, common + 'rw=write\nbs=1M\nverify=crc32c\n'
                'do_verify=1\nverify_fatal=1\nrefill_buffers=1\nend_fsync=1\n[prepare]\n',
                args.output, 'prepare', readonly=False)
        stat = file.stat()
        assert stat.st_size == size and stat.st_blocks * 512 >= size
        (args.output / 'file-allocation.json').write_text(json.dumps(
            {'size': stat.st_size, 'allocated_bytes': stat.st_blocks * 512, 'device': stat.st_dev}))
        result = subprocess.run(['filefrag', '-v', str(file)], capture_output=True, text=True)
        (args.output / 'filefrag.txt').write_text(result.stdout + result.stderr)
        for repeat in range(1, 4):
            # Interleave the four load types to expose drift across rounds.
            for mode, bs, jobs in [('read', '1M', 1), ('randread', '4k', 1),
                                   ('read', '1M', 4), ('randread', '4k', 4)]:
                cfg = common + ('rw=' + mode + '\nbs=' + bs + '\nnumjobs=' + str(jobs) +
                                '\ntime_based=1\nruntime=15\nramp_time=3\n'
                                'allow_file_create=0\nrandrepeat=1\nrandseed=20261013\n[read]\n')
                name = mode + '-' + bs + '-j' + str(jobs) + '-r' + str(repeat)
                run_fio(args.fio, cfg, args.output, name)
    finally:
        # Only the uniquely created file belongs to this invocation.
        if file.exists():
            file.unlink()
        (args.output / 'cleanup.json').write_text(json.dumps({'test_file_removed': not file.exists()}))


if __name__ == '__main__':
    main()
