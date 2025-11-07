#!/usr/bin/env python3
import os, json, time, requests
from pathlib import Path
ART_DIR = Path(os.getenv("ART_DIR", "\tmp\art"))
LOG = ART_DIR / "gateway_admin_ops.log"
BASE = os.getenv("GATEWAY_BASE", "http://127.0.0.1:9910")
TOK = os.getenv("GATEWAY_ADMIN_TOKEN", "").strip()
def _headers():
    h = {"content-type": "application/json"}
    if TOK:
        h["x-admin-token"] = TOK
    return h
def replay(n=20):
    if LOG.exists():
        lines = LOG.read_text(encoding="utf-8").splitlines()[-n:]
        print(f"[replay] found {len(lines)} admin events, but this script will send synthetic /answer probes instead")
    qs = [
        "what is the gateway topology?",
        "show me the latency profile",
        "explain RAG fan-out policies",
    ]
    for q in qs:
        try:
            r = requests.post(f"{BASE}/answer", json={"query": q}, headers=_headers(), timeout=3.0)
            print("[replay]", q, "→", r.status_code)
        except Exception as e:
            print("[replay] error:", e)
if __name__ == "__main__":
    replay()
