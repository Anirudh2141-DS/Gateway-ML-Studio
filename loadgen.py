#!/usr/bin/env python3
import os, time, json, argparse, random, threading, queue, httpx
from prometheus_client import start_http_server, Histogram, Counter
LAT = Histogram("loadgen_latency_ms", "Loadgen latency (ms)", ["scenario"])
ERR = Counter("loadgen_errors_total", "Loadgen errors", ["scenario"])
def worker(q, base, ns, scenario):
    with httpx.Client(timeout=3.0) as c:
        while True:
            try:
                _ = q.get(timeout=1)
            except queue.Empty:
                return
            t0 = time.time()
            try:
                r = c.post(f"{base}/answer", json={"query": "test query " + str(random.randint(1,9)), "top_k": 3},
                           headers={"X-Namespace": ns})
                r.raise_for_status()
                dt = (time.time()-t0)*1000.0
                LAT.labels(scenario=scenario).observe(dt)
            except Exception:
                ERR.labels(scenario=scenario).inc()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.getenv("GATEWAY_BASE", "http://127.0.0.1:9910"))
    ap.add_argument("--ns", default="default")
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--qps", type=float, default=8.0)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--metrics-port", type=int, default=9108)
    args = ap.parse_args()
    start_http_server(args.metrics_port)
    total = int(args.qps * args.seconds)
    q = queue.Queue()
    for _ in range(total): q.put(1)
    threads = []
    for _ in range(max(1,int(args.qps))):
        t = threading.Thread(target=worker, args=(q, args.gateway, args.ns, args.scenario), daemon=True)
        t.start(); threads.append(t)
    time.sleep(args.seconds + 2)
    print(json.dumps({"sent": total, "scenario": args.scenario}))
if __name__ == "__main__":
    main()
