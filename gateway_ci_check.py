#!/usr/bin/env python3
import json, sys, argparse
from pathlib import Path
ART_DIR = Path("\tmp\art")
agg_path = ART_DIR / "gateway_runs_agg.json"
schema_path = ART_DIR / "gateway_schema_report.json"
drift_path = ART_DIR / "gateway_drift_report.json"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-p95", type=float, default=1800.0)
    args = ap.parse_args()
    # 1) schema must not have problems (if file exists)
    if schema_path.exists():
        rep = json.loads(schema_path.read_text())
        probs = rep.get("problems") or []
        if probs:
            print("[ci] ❌ schema problems:", probs)
            sys.exit(1)
    else:
        print("[ci] ⚠ no schema report, continuing")
    # 2) p95 must be under threshold
    if agg_path.exists():
        agg = json.loads(agg_path.read_text())
        runs = agg.get("runs") or []
        if runs:
            latest = runs[0]
            p95 = latest.get("p95_ms")
            if isinstance(p95, (int, float)) and p95 > args.max_p95:
                print(f"[ci] ❌ p95 too high: {p95} ms > {args.max_p95} ms")
                sys.exit(1)
            else:
                print(f"[ci] ✅ p95 ok: {p95} ms")
        else:
            print("[ci] ⚠ agg has no runs, continuing")
    else:
        print("[ci] ⚠ no agg file, continuing")
    # 3) drift detection is non-fatal
    if drift_path.exists():
        d = json.loads(drift_path.read_text())
        if d.get("chaos_drift") or d.get("rl_drift"):
            print("[ci] ⚠ drift detected (non-fatal):", d)
        else:
            print("[ci] ✅ no drift")
    print("[ci] ✅ all checks passed")
    sys.exit(0)
if __name__ == "__main__":
    main()
