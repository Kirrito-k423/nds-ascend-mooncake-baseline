# RMB Cost

- Generated at: `2026-10-13T11:35:46Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: MAC-17
- Session: runtime session (local path omitted)
- Token window events: `None` -> `2026-10-13T11:34:35.580Z`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 33,608,967 | 33.608967 |
| Cached input | 33,025,664 | 33.025664 |
| Uncached input | 583,303 | 0.583303 |
| Output | 167,075 | 0.167075 |
| Reasoning output, included in output when provider reports it that way | 77,299 | 0.077299 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| GPT reference pricing scenario | 5 | 0.5 | 30 | Unverified skill defaults; not actual runtime invoice |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Unverified default comparison scenario |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| GPT reference pricing scenario | Input (cache miss) | not cached | 583,303 | 0.583303 | 5 | 2.92 | 21.00 |
| GPT reference pricing scenario | Input (cache hit) | cached | 33,025,664 | 33.025664 | 0.5 | 16.51 | 118.89 |
| GPT reference pricing scenario | Output | not applicable | 167,075 | 0.167075 | 30 | 5.01 | 36.09 |
| deepseek-v4-pro | Input (cache miss) | not cached | 583,303 | 0.583303 | 0.435 | 0.2537 | 1.83 |
| deepseek-v4-pro | Input (cache hit) | cached | 33,025,664 | 33.025664 | 0.003625 | 0.1197 | 0.8620 |
| deepseek-v4-pro | Output | not applicable | 167,075 | 0.167075 | 0.87 | 0.1454 | 1.05 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| GPT reference pricing scenario | 21.00 | 118.89 | 36.09 | 175.98 | 24.44 |
| deepseek-v4-pro | 1.83 | 0.8620 | 1.05 | 3.74 | 0.5188 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
