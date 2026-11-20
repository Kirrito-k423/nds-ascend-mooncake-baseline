"""Require four idle cards; A5 Warning is allowed when compute tests pass."""
import json
from pathlib import Path
import re
import sys

if __name__=='__main__':
    text=Path(sys.argv[1]).read_text()
    selected=[int(x) for x in sys.argv[2].split(',')]
    assert len(selected)==4 and len(set(selected))==4
    health={int(m[1]):m[2] for m in re.finditer(
        r'^\|\s*(\d+)\s*\|\s*\S+\s*\|\s*(OK|Warning|Alarm)\s*\|',text,re.M)}
    idle=set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)\b',text)))
    result={'selected':selected,'health':health,'idle':sorted(idle),'warning_allowed':True,
            'pass':all(health.get(x) in ['OK','Warning'] and x in idle for x in selected)}
    Path('device-gate.json').write_text(json.dumps(result,indent=2))
    if not result['pass']:raise SystemExit('Selected card occupied, absent, or Alarm; preserve other work')
