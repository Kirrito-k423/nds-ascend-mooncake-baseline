"""Record SSE arrivals, including empty events and actual usage, without retries."""
import json
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(base, tokens, timeout=600):
    body = {'model': 'qwen', 'prompt': tokens, 'max_tokens': 32, 'ignore_eos': True,
            'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True},
            'logprobs': 1}
    begin = time.perf_counter_ns()
    result = {'start_ns': begin, 'expected_prompt_tokens': len(tokens), 'events': [], 'success': False}
    req = urllib.request.Request(base+'/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with OPENER.open(req, timeout=timeout) as response:
            for raw in response:
                when = time.perf_counter_ns()
                if not raw.startswith(b'data: '): continue
                value = raw[6:].strip()
                if value == b'[DONE]': result['done'] = True; break
                event = json.loads(value)
                result['events'].append({'arrival_ns': when, 'event': event})
                if event.get('usage'): result['usage'] = event['usage']
        usage = result.get('usage', {})
        result['success'] = (result.get('done') is True and usage.get('prompt_tokens') == len(tokens)
                             and usage.get('completion_tokens') == 32)
        if not result['success']: result['error'] = 'Usage or DONE mismatch'
    except Exception as e:
        result['error'] = type(e).__name__+': '+str(e)
    result['finish_ns'] = time.perf_counter_ns()
    # logprobs count emitted tokens, even if their decoded text is empty.
    arrivals = []
    event_token_counts = []
    for item in result['events']:
        count = sum(len((choice.get('logprobs') or {}).get('tokens') or [])
                    for choice in item['event'].get('choices', []))
        if count:
            event_token_counts.append(count)
            arrivals.extend([item['arrival_ns']]*count)
    result['token_arrivals_ns'] = arrivals
    result['event_token_counts'] = event_token_counts
    result['token_timing_exact'] = len(arrivals) == 32 and all(n == 1 for n in event_token_counts)
    if len(arrivals) == 32:
        result['ttft_ms'] = (arrivals[0]-begin)/1e6
        result['tpot_ms'] = (arrivals[-1]-arrivals[0])/31e6
        result['token_intervals_ms'] = [(b-a)/1e6 for a,b in zip(arrivals,arrivals[1:])]
    return result
