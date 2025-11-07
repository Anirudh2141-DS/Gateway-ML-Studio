#!/usr/bin/env python3
import sys, os, json, argparse, time
import urllib.request
def q(url, query):
    u = f"{url}?query={urllib.parse.quote(query)}"
    with urllib.request.urlopen(u, timeout=5) as r:
        return json.loads(r.read().decode())
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default=os.getenv("PROM_QUERY_URL","http://localhost:9090/api/v1/query"))
    ap.add_argument("--p95", type=float, default=1.8, help="p95 seconds")
    ap.add_argument("--golden", type=float, default=95.0, help="golden pass %")
    args = ap.parse_args()
    p95q = 'histogram_quantile(0.95, sum(rate(gateway_answer_latency_ms_bucket[10m])) by (le))'
    j = q(args.prom, p95q)
    try:
        val = float(j["data"]["result"][0]["value"][1])
    except Exception:
        print("WARN: no p95 data; failing gate"); sys.exit(2)
    print(f"p95={val:.3f}s (limit {args.p95:.3f}s)")
    if val > args.p95:
        print("FAIL: p95 too high"); sys.exit(1)
    # Golden pass placeholder (wire to your golden job if exported to Prometheus)
    print(f"golden>={args.golden}% assumed pass (placeholder)")
    sys.exit(0)
if __name__ == "__main__":
    main()
