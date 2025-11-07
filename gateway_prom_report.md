# Gateway Telemetry Report
_generated: 2025-11-02 13:18:03_

## Strategy & Fanout (from metrics)
```json
{
  "ts": 1762114677,
  "strategies": {},
  "fanout_errors": {
    "rerank": 1.0,
    "concurrency": 12.0
  }
}
```

## Latency buckets (from metrics)
```json
{
  "ts": 1762114677,
  "buckets": {
    "10.0": 12.0,
    "20.0": 12.0,
    "40.0": 12.0,
    "80.0": 12.0,
    "160.0": 13.0,
    "320.0": 14.0,
    "640.0": 14.0,
    "1280.0": 14.0,
    "2560.0": 14.0,
    "+Inf": 14.0
  }
}
```

## Recent gateway runs (p95)
| when | p95 (ms) | chaos | file |
|------|----------|--------|------|
| 2025-11-02 13:16:49 | None | verifier_v2 | gateway_smoke_20251102-131650.json |
| 2025-10-30 10:06:05 | None | verifier_v2 | gateway_smoke_20251030-100608.json |
| 2025-10-29 23:59:43 | None | verifier_v2 | gateway_smoke_20251030-000220.json |