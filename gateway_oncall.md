# Gateway On-Call

## 0. Ping
- `curl -s http://127.0.0.1:9910/up | jq`
- `curl -s http://127.0.0.1:9910/metrics | head -200`

## 1. Check chaos
- `curl -s http://127.0.0.1:9910/admin/chaos/get | jq`

## 2. Force BM25 (dense broken)
- `curl -s -X POST http://127.0.0.1:9910/admin/chaos/set -H 'content-type: application/json' -d '{"force_bm25":true}' | jq`

## 3. Disable CE (rerank bloating latency)
- `curl -s -X POST http://127.0.0.1:9910/admin/chaos/set -H 'content-type: application/json' -d '{"disable_ce":true}' | jq`

## 4. See audit log (who touched it)
- `cat \tmp\art\gateway_audit.jsonl`

## 5. Re-apply desired state
- open notebook and run `auto_remediate_drift(apply=True)`