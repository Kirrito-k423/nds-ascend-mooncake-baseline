"""Create a reviewed text-only evidence archive; exclude connection inventory."""
import argparse
import hashlib
import ipaddress
import json
import re
import shutil
import zipfile
from pathlib import Path

SKIP_NAMES={'containers-before.txt','containers-after.txt','hostname.txt','memory.txt','disk.txt','docker-images.txt'}
SKIP_TASKS={'inventory-b078','inventory-b021'}

def redact(text):
    def node(match):
        try: ip=ipaddress.ip_address(match[0])
        except ValueError:return match[0]
        return '<NODE-IP>' if ip.is_private and not ip.is_loopback and not ip.is_unspecified else match[0]
    text=re.sub(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',node,text)
    text=re.sub(r'/root/\.simplehtmlwatch/tasks/[\w-]+/results','<TASK_RESULTS>',text)
    text=re.sub(r'/home/[\w-]+/mac17-v4flash-[\w-]+','<PROJECT_DIR>',text)
    text=re.sub(r'/Users/[^\s"\n]+','<LOCAL_PATH>',text)
    # Kernel process names and our worker PIDs are evidence; unrelated inventory is omitted.
    return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);a=p.parse_args()
    if a.dest.exists():shutil.rmtree(a.dest)
    a.dest.mkdir(parents=True);entries=[]
    for source in sorted(a.source.rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(a.source)
        if source.name in SKIP_NAMES or any(part in SKIP_TASKS for part in relative.parts):continue
        if '__pycache__' in relative.parts or source.suffix in {'.pyc','.safetensors'}:continue
        # Keep public source snapshots unchanged so the runtime/source hashes remain useful.
        public=relative.parts[0] in {'sources','model-metadata'} or 'installed-source' in relative.parts or 'runtime-snapshot' in relative.parts
        body=source.read_bytes()
        if not public:
            body=redact(body.decode('utf-8')).encode('utf-8')
        target=a.dest/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
        entries.append({'path':str(relative),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),
                        'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'redacted':body!=source.read_bytes()})
    notice='''MAC-17 evidence. Public metadata/source snapshots retain original bytes and source licenses.
Runtime logs retain measured results; private node IPs and project paths are replaced by markers.
Unrelated container inventory and other machines' process lists are omitted.
Headers contain metadata only; no real weight tensor payload is included.
Source code derived from vLLM and vLLM-Ascend is subject to its original Apache-2.0 license.
The archive contains successful and failed runs; use result.json and the report to distinguish them.
'''
    target=a.dest/'NOTICE.txt';target.write_text(notice)
    entries.append({'path':'NOTICE.txt','bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    (a.dest/'SHA256.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
    a.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(a.dest.rglob('*')):
            if path.is_file():archive.write(path,Path('evidence/mac17')/path.relative_to(a.dest))
    print(json.dumps({'files':len(entries),'archive_bytes':a.zip.stat().st_size,
                      'archive_sha256':hashlib.sha256(a.zip.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
