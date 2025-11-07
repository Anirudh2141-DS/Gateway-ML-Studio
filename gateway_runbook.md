# RAG Gateway Runbook (auto-generated)

## 1. Check live vs control
```python
diff_control_vs_live()
```

## 2. Force safemode
```python
apply_safemode(reason='latency-spike')
```

## 3. Clear safemode (back to desired)
```python
clear_safemode()
```

## 4. Replay failed ops
```python
from pathlib import Path
# previous cell defined replay_pending_ops()
replay_pending_ops()  # best-effort
```

## 5. Rollback control-plane
```python
rollback_control_plane(1)
push_current_control_plane()
```