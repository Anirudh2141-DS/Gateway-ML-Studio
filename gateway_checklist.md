# Gateway Checklist

1. **Health**
   - `curl -s http://127.0.0.1:9910/up | jq`
   - `curl -s http://127.0.0.1:9910/metrics | head -200`

2. **Admin**
   - `curl -s http://127.0.0.1:9910/admin/chaos/get | jq`  # (no token set)

3. **Smoke**
   - `curl -s -X POST http://127.0.0.1:9910/answer -H 'content-type: application/json' -d '{"query":"smoke","top_k":2}' | jq`

4. **Perf (optional)**
   - run notebook `loadgen(...)` or CLI bench

5. **Metrics sanity**
   - `curl -s http://127.0.0.1:9910/metrics | grep gateway_answer_latency_ms_bucket | head -20`