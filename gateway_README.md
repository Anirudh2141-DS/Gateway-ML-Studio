# Gateway Control Plane (Notebook Build)

- Host: `LAPTOP-GV10237H`
- Env: `dev`
- Generated: Sun Nov  2 14:52:25 2025

## Quickstart
```bash
python notebook.py  # your jupyter-run
gw_cli15('boot')     # or gw_cli16('boot.secure')
```

## HTTP Ports

## Security
- require_admin_token: False
- require_mtls: False

## Useful CLI
- gw_cli9('freeze', reason='incident')
- gw_cli12('promote', reason='deploy')
- gw_cli13('metrics')
- gw_cli14('timeline')
- gw_cli15('slo.eval')
- gw_cli16('snapshot')