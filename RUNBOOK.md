# RUNBOOK
## Env
- RAG_BASE, GATEWAY_PORT, OVERSEER_PORT
- Chaos toggles via `/admin/chaos/set`: `disable_ce`, `force_bm25`, `verifier_v2`
## Health
- Gateway: `/up`, `/metrics`, `/admin/health`
- RAG: `/up`, `/metrics` (prom wrapper), `/complete`
## Canary
- Update: `POST /rl/update {"weights":{"factual":0.1}}`
- Promote snapshot: `POST /rl/promote`
- Rollback: `POST /rl/rollback`
- Status: `GET /canary/status`, `GET /canary/strategy_counts`
## Docker
- `docker compose up -d --build`
## Gates
- `python scripts/metrics_check.py --prom http://<prom>/api/v1/query`
## Loadgen
- `python scripts/loadgen.py --gateway http://127.0.0.1:9910 --qps 12 --seconds 60`
