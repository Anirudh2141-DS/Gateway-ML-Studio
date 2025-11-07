# Gateway Ops
_generated: 2025-11-02 13:18:08_

## Health
- `curl -s http://127.0.0.1:9910/up | jq`
- `curl -s http://127.0.0.1:9910/metrics | head -200`

## Chaos
- `curl -s http://127.0.0.1:9910/admin/chaos/get | jq`
- `curl -s -X POST http://127.0.0.1:9910/admin/chaos/set -H 'content-type: application/json' -d '{"disable_ce":true}' | jq`

## RL Weights
- `curl -s -X POST http://127.0.0.1:9910/rl/promote -H 'content-type: application/json' -d '{}' | jq`
- `curl -s -X POST http://127.0.0.1:9910/rl/rollback -H 'content-type: application/json' -d '{}' | jq`

## Strategy Mix
- `curl -s http://127.0.0.1:9910/canary/strategy_counts | jq`

## Emergency (slow / dense errors)
- `curl -s -X POST http://127.0.0.1:9910/admin/chaos/set -H 'content-type: application/json' -d '{"force_bm25":true}' | jq`