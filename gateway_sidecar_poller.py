#!/usr/bin/env python3
import os, json, time, requests
from pathlib import Path
ART_DIR = Path(os.getenv("ART_DIR", "\tmp\art"))
CONTROL = ART_DIR / "gateway_control.json"
GATEWAY_BASE = os.getenv("GATEWAY_BASE", "http://127.0.0.1:9910")
ADMIN_TOKEN = os.getenv("GATEWAY_ADMIN_TOKEN", "").strip()
INTERVAL = float(os.getenv("GW_SIDECAR_INTERVAL", "5"))

def _headers():
    h = {"content-type": "application/json"}
    if ADMIN_TOKEN:
        h["x-admin-token"] = ADMIN_TOKEN
    return h
def main():
    last_seen = None
    print(f"[sidecar] watching {CONTROL} every {INTERVAL}s → {GATEWAY_BASE}")
    while True:
        try:
            if CONTROL.exists():
                txt = CONTROL.read_text(encoding="utf-8")
                if txt != last_seen:
                    # push global chaos
                    doc = json.loads(txt)
                    chaos = doc.get("chaos") or {}
                    rl = doc.get("rl_weights") or {}
                    print("[sidecar] detected change → pushing chaos+rl")
                    r1 = requests.post(f"{GATEWAY_BASE}/admin/chaos/set", headers=_headers(), json=chaos, timeout=2.5)
                    r2 = requests.post(f"{GATEWAY_BASE}/rl/update", headers=_headers(), json={"weights": rl}, timeout=2.5)
                    print("[sidecar] chaos:", r1.status_code, "rl:", r2.status_code)
                    last_seen = txt
            else:
                print("[sidecar] control-file missing, skipping")
        except Exception as e:
            print("[sidecar] error:", e)
        time.sleep(INTERVAL)
if __name__ == "__main__":
    main()
