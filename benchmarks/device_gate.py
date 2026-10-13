"""Allow A5 Warning; require the selected cards to be idle before validation."""
import json, pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text()
health = {int(m[1]):m[2] for m in re.finditer(
    r'^\|\s*(\d+)\s*\|\s*\S+\s*\|\s*(OK|Warning|Alarm)\s*\|', text, re.M)}
selected = [int(i) for i in sys.argv[2].split(',')]
idle = set(int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)\b', text))
report = {'selected_cards':selected, 'health':health,
          'warning_allowed':True, 'idle_cards':sorted(idle)}
pathlib.Path('device-gate.json').write_text(json.dumps(report, indent=2))
if any(health.get(i) not in ('OK','Warning') or i not in idle for i in selected):
    raise SystemExit('Selected card missing, occupied, or needs separate Alarm validation')
print(json.dumps(report, indent=2))
