# Gateway Ops (Notebook Edition)

## 1. Alert checks (cooldown)
```python
alert_on_agg_with_cooldown(1800.0, 900)
```

## 2. Sync staging/EU from prod
```python
sync_gateways_from_source('http://127.0.0.1:9910')
```

## 3. Hot-reload
```python
notebook_hot_reload_loop(interval_s=5.0, max_loops=100)
```

## 4. Grafana
```python
write_grafana_dashboard()
```