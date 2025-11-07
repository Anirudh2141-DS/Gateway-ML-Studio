# auto-generated CLI
import sys, json, os, time
from pathlib import Path

ART_DIR = Path(r"\tmp\art")
sys.path.insert(0, str(ART_DIR))
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "version"
    if cmd == "version":
        info_path = ART_DIR / "gateway_controller_info.json"
        if info_path.exists():
            print(info_path.read_text())
        else:
            print(json.dumps({"error":"no controller info"}, indent=2))
    elif cmd == "maintenance":
        # will be no-op if not in scope
        try:
            from __main__ import controller_maintenance
            res = controller_maintenance()
            print(json.dumps(res, indent=2))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    elif cmd == "slo.eval":
        try:
            from __main__ import eval_slos
            p = eval_slos()
            print(Path(p).read_text())
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        print("unknown cmd:", cmd)
if __name__ == "__main__":
    main()
