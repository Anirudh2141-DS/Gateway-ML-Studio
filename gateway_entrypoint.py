#!/usr/bin/env python3
# auto-generated: notebook entrypoint
import json, os, sys, time
from pathlib import Path
ART_DIR = Path("\tmp\art")
E2E = ART_DIR / "gateway_e2e_report.json"
DISK = ART_DIR / "gateway_disk_alert.json"
SCEN = ART_DIR / "gateway_scenarios.json"
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "e2e"
    if cmd == "e2e":
        from __main__ import e2e_probe
        e2e_probe()
    elif cmd == "disk":
        from __main__ import disk_alert
        disk_alert()
    elif cmd == "scenarios":
        from __main__ import run_scenarios
        run_scenarios()
    else:
        print("unknown cmd:", cmd)
if __name__ == "__main__":
    main()
