# Gateway Runbook (auto-generated)

- Generated: Sun Nov  2 13:35:31 2025
- Host: `LAPTOP-GV10237H`

## Freeze / Incident
- Freeze gateway: `gw_cli9('freeze', reason='incident')`
- Unfreeze gateway: `gw_cli9('unfreeze')`
- Check: `GET /healthz` on n8n-http

## Promotion
- Run CI gate: `gw_cli9('ci.gate')`
- If blocked → check `gateway_promotion_gate.json` and `gateway_canary_eval.json`
- If OK → switch blue/green via controller HTTP

## DLQ
- Inspect: `cat /tmp/art/n8n-dlq.jsonl`
- Replay: `gw_cli10('dlq.replay')` (this file)

## Drift
- Run: `gw_cli10('drift')`
- See: `/tmp/art/gateway_drift_report.json`

## Notifications
- Slack webhook is read from `vault-secrets.json` key: `slack/webhook`
- Fallback log: `/tmp/art/gateway_notify.jsonl`