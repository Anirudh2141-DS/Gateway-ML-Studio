# Gateway Notebook Targets

```bash
# full daily run
python -c "from notebook import gw_run; gw_run('daily')"

# refresh telemetry only
python -c "from notebook import dump_all_telemetry; dump_all_telemetry()"

# expose artifacts locally
python -c "from notebook import serve_artifacts; serve_artifacts(9919)"
```