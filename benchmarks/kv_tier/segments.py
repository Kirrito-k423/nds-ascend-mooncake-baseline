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
