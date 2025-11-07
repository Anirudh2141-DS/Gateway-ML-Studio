# Gateway GitOps bundle (auto-generated)
apply in this order:
1. prometheus-gateway-rules.yml (if you manage Prom separately, skip)
2. k8s-gateway-sidecar.yaml
3. k8s-gateway-agent-cronjob.yaml
4. helm-gateway-values.yaml (for Helm-based deploys)
