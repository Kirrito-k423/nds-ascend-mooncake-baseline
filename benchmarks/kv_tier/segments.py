"""Select contributing SDK control hostnames from successful initialization."""
import json

def contributing_segments(text,host,expected=4):
    names=set()
    for line in text.splitlines():
        if 'NDS_TRACE ' not in line:continue
        row=json.loads(line.split('NDS_TRACE ',1)[1])
        if row.get('op')!='store_ready' or not row.get('contributes') or row.get('pool_bytes',0)<=0:continue
        name=row['segment_name']
        assert isinstance(name,str) and name.startswith(host+':'),'Unexpected contributing node'
        names.add(name)
    assert len(names)==expected,'Missing or ambiguous contributing SDK control hostnames'
    return sorted(names)


def contributing_controls(text,host,expected=4):
    controls={}
    for line in text.splitlines():
        if 'NDS_TRACE ' not in line:continue
        row=json.loads(line.split('NDS_TRACE ',1)[1])
        if row.get('op')!='store_ready' or not row.get('contributes') or row.get('pool_bytes',0)<=0:continue
        name=row['segment_name'];path=row.get('control_path')
        assert name.startswith(host+':') and isinstance(path,str) and path.startswith('/evidence/disk/worker-clear-')
        assert row.get('worker_pid',0)>0
        if name in controls:assert controls[name]==path,'Ambiguous worker control path'
        controls[name]=path
    assert len(controls)==expected,'Require all contributing original worker clients'
    return controls
