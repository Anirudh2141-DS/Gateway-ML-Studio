# Gateway Notebook Docs (auto-generated)

- Generated at: Sun Nov  2 13:19:15 2025
- Host: `LAPTOP-GV10237H`

## Artifacts
- Total artifacts: **148**
  - `.github/workflows/gateway-notebook-ci.yml` (879 bytes)
  - `backups/art-backup-1761861720.tar.gz` (68571 bytes)
  - `bundle/.github/workflows/ci.yml` (986 bytes)
  - `bundle/dashboards/gateway_rag_overview.json` (1078 bytes)
  - `bundle/docker-compose.yml` (630 bytes)
  - `bundle/Dockerfile.gateway` (544 bytes)
  - `bundle/Dockerfile.microservice` (296 bytes)
  - `bundle/gateway/Cargo.toml` (549 bytes)
  - `bundle/gateway/src/main.rs` (847 bytes)
  - `bundle/MANIFEST.json` (2744 bytes)
  - `bundle/model_cards/BGE_CE.md` (135 bytes)
  - `bundle/model_cards/E5_Base.md` (150 bytes)
  - `bundle/model_cards/MNLI_RoBERTa.md` (144 bytes)
  - `bundle/openapi_microservice.json` (4974 bytes)
  - `bundle/overseer.py` (81 bytes)
  - `bundle/rag_service.py` (127 bytes)
  - `bundle/RUNBOOK.md` (682 bytes)
  - `bundle/scripts/loadgen.py` (1905 bytes)
  - `bundle/scripts/metrics_check.py` (1205 bytes)
  - `control-history/control-1761848359.json` (1056 bytes)
  - `control-history/control-1762114724.json` (1056 bytes)
  - `deploy/gateway_runbook.md` (522 bytes)
  - `deploy/helm-gateway-values.yaml` (375 bytes)
  - `deploy/k8s-gateway-agent-cronjob.yaml` (2061 bytes)
  - `deploy/k8s-gateway-sidecar.yaml` (1036 bytes)
  - `deploy/prometheus-gateway-rules.yml` (1035 bytes)
  - `deploy/README.md` (253 bytes)
  - `docker-compose.gateway-notebook.yml` (564 bytes)
  - `docker-compose.gateway.yml` (391 bytes)
  - `Dockerfile.gateway` (412 bytes)
  - `Dockerfile.gateway-notebook` (323 bytes)
  - `fixtures/control.json` (454 bytes)
  - `fixtures/prom_ts.json` (126 bytes)
  - `fixtures/runs_agg.json` (226 bytes)
  - `fixtures/schema_report.json` (22 bytes)
  - `gateway-ci.yml` (958 bytes)
  - `gateway-values.yaml` (305 bytes)
  - `gateway_admin.rs` (3455 bytes)
  - `gateway_agent.sh` (487 bytes)
  - `gateway_artifact_index.json` (21482 bytes)

## Lint
- ✅ State lint: OK

## SLOs
### Defined
```json
{
  "ts": 1762114736,
  "schema": "1.1.0",
  "global": {
    "p95_ms": 1800.0,
    "error_rate": 0.05
  },
  "tenants": {
    "default": {
      "p95_ms": 1800.0,
      "error_rate": 0.05
    },
    "tenant-acme": {
      "p95_ms": 1200.0,
      "error_rate": 0.03
    },
    "tenant-beta": {
      "p95_ms": 1500.0,
      "error_rate": 0.04
    }
  }
}
```
### Last Evaluation
```json
{
  "ts": 1762114736,
  "schema": "1.1.0",
  "global": {
    "p95_ms": 1800.0,
    "error_rate": 0.05
  },
  "tenants": {
    "default": {
      "tenant": "default",
      "target_p95_ms": 1800.0,
      "actual_p95_ms": 3.7,
      "target_error_rate": 0.05,
      "actual_error_rate": 1.0,
      "total_probes": 2,
      "breach_p95": false,
      "breach_error": true
    },
    "tenant-acme": {
      "tenant": "tenant-acme",
      "target_p95_ms": 1200.0,
      "actual_p95_ms": 5.0,
      "target_error_rate": 0.03,
      "actual_error_rate": 1.0,
      "total_probes": 2,
      "breach_p95": false,
      "breach_error": true
    },
    "tenant-beta": {
      "tenant": "tenant-beta",
      "target_p95_ms": 1500.0,
      "actual_p95_ms": 4.8,
      "target_error_rate": 0.04,
      "actual_error_rate": 1.0,
      "total_probes": 2,
      "breach_p95": false,
      "breach_error": true
    }
  },
  "breaches": [
    {
      "tenant": "default",
      "target_p95_ms": 1800.0,
      "actual_p95_ms": 3.7,
      "target_error_rate": 0.05,
      "actual_error_rate": 1.0,
      "total_probes": 2,
      "breach_p95": false,
      "breach_error": true
    },
    {
      "tenant": "tenant-acme",
      "target_p95_ms": 1200.0,
      "actual_p95_ms": 5.0,
      "target_error_rate": 0.03,
      "actual_error_rate": 1.0,
      "total_probes": 2,
      "breach_p95": false,
      "breach_error": true
    },
    {
      "tenant": "tenant-beta",
      "target_p95_ms": 1500.0,
      "actual_p95_ms": 4.8,
      "target_error_rate": 0.04,
      "actual_error_rate": 1.0,
      "total_probes": 2,
      "breach_p95": false,
      "breach_error": true
    }
  ],
  "sources": {
    "e2e": "gateway_e2e_report.json",
    "sla": "gateway_sla_runs.json",
    "prom_ts": "gateway_prom_timeseries.json"
  }
}
```