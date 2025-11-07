#!/usr/bin/env python3
import json, os, time
from pathlib import Path
ART_DIR = Path(os.getenv("ART_DIR", "\tmp\art"))

def main():
    # this imports the notebook-namespace funcs if run inside same env
    try:
        from __main__ import notebook_bootstrap
    except Exception:
        notebook_bootstrap = None
    steps = []
    if notebook_bootstrap:
        steps.append("bootstrap")
        notebook_bootstrap("dev")
    # SLOs
    try:
        from __main__ import write_default_slos, eval_slos
        write_default_slos()
        eval_slos()
        steps.append("slo")
    except Exception:
        pass
    # burn
    try:
        from __main__ import compute_burn_rates
        compute_burn_rates()
        steps.append("burn")
    except Exception:
        pass
    # gitops
    try:
        from __main__ import collect_gitops
        collect_gitops()
        steps.append("gitops")
    except Exception:
        pass
    out = {
        "ts": int(time.time()),
        "steps": steps,
        "art_dir": str(ART_DIR),
    }
    (ART_DIR / "gateway_quickstart_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[quickstart] done:", steps)
if __name__ == "__main__":
    main()
