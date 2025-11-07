# app.py — Gateway ML Studio (ULTRA-LITE v1.1, FIXED)
# Run: streamlit run app.py

import os, io, json, time, shutil, secrets, hashlib, contextlib, re, gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Page + Lite settings
# -------------------------
st.set_page_config(page_title="Gateway ML Studio (ULTRA-LITE)", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

# ULTRA-LITE knobs
LITE_MODE = True
ROW_CAP_TRAIN = 5000           # hard cap for training sample
PREVIEW_ROWS = 200             # preview size
CHUNK_SIZE_PROFILE = 20000     # streamed profiling chunk size
ENABLE_ENSEMBLES = True        # allow tiny ensembles on demand
ENABLE_XGB = False             # keep off on 8GB
PERM_IMPORTANCE = False        # off in ULTRA-LITE
HAVE_FORECAST = False          # off in ULTRA-LITE

# -------------------------
# Paths
# -------------------------
ART_DIR = Path(os.getenv("ART_DIR", Path.cwd() / "gateway_artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_ROOT = ART_DIR / "sessions"; SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

def _rm_tree(p: Path):
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass

if "session_id" not in st.session_state:
    st.session_state.session_id = secrets.token_hex(6)
    # single-user desktop: purge old sessions
    for d in SESSIONS_ROOT.iterdir():
        if d.is_dir() and d.name != st.session_state.session_id:
            _rm_tree(d)

SESSION_DIR = SESSIONS_ROOT / st.session_state.session_id
DATA_DIR = SESSION_DIR / "data"
MODEL_DIR = SESSION_DIR / "model_registry"
for d in (DATA_DIR, MODEL_DIR): d.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("End & purge session", use_container_width=True):
        _rm_tree(SESSION_DIR); st.success("Purged. Reloading…"); st.rerun()

# -------------------------
# Minimal helpers
# -------------------------
def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

def _write_gateway_file(name: str, obj: Dict[str, Any]):
    save_json(obj, SESSION_DIR / name)
    save_json(obj, ART_DIR / name)

def _bytes_h(n:int)->str:
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def _clean_columns(df: pd.DataFrame)->pd.DataFrame:
    out = df.copy()
    try:
        # Avoid duplicate column names causing pipeline issues
        base = pd.io.parsers.ParserBase({'names': out.columns})
        out.columns = base._maybe_dedup_names(out.columns)  # may break in future pandas; safe today
    except Exception:
        pass
    out.columns = [str(c).strip().replace("\n"," ").replace("\r"," ") for c in out.columns]
    return out

# ---------- robust numeric coercion (handles '-', currency, commas, () negatives, spaces, weird dashes)
_NUM_BADS = {
    "", " ", "-", "—", "–", "- ", " -", " - ", "NA", "N/A", "na", "n/a", "None", "null", ".", ".."
}
_CURRENCY = re.compile(r"[,$£€₹\s]")  # strip common currency/commas/spaces
_PARENS_NEG = re.compile(r"^\(\s*([^)]+)\s*\)$")

def _to_float_smart(x) -> Optional[float]:
    if x is None: return np.nan
    if isinstance(x, (int, float, np.number)):
        # keep as is
        return float(x)
    s = str(x).strip()
    if s in _NUM_BADS: return np.nan
    # parentheses negatives e.g. "(1234.50)" -> -1234.50
    m = _PARENS_NEG.match(s)
    if m: s = "-" + m.group(1)
    # remove currency, commas, spaces
    s = _CURRENCY.sub("", s)
    # replace weird unicode dashes with minus
    s = s.replace("—","-").replace("–","-")
    try:
        return float(s)
    except Exception:
        return np.nan

def _coerce_light(df: pd.DataFrame)->pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_object_dtype(s):
            # Try numeric first using smart parser on a small sample; if >60% parse succeeds, convert all
            sample = s.dropna().astype(str).head(50).tolist()
            if sample:
                parses = sum(pd.notna([_to_float_smart(v) for v in sample]))
                if parses >= max(10, int(0.6*len(sample))):
                    out[c] = s.map(_to_float_smart)
                    continue
            # Try datetime if many date-like tokens
            with contextlib.suppress(Exception):
                sm = s.dropna().astype(str).head(50)
                if sum(any(ch in v for ch in "-/:") for v in sm) >= max(5, int(0.2*len(sm))):
                    out[c] = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    # downcast numerics
    with contextlib.suppress(Exception):
        for c in out.select_dtypes(include=["float64","int64"]).columns:
            out[c] = pd.to_numeric(out[c], errors="coerce", downcast="float")
        for c in out.select_dtypes(include=["int64"]).columns:
            out[c] = pd.to_numeric(out[c], errors="coerce", downcast="integer")
    return out

# -------------------------
# Robust CSV opener with fallback encodings
# -------------------------
def _read_csv_try_encodings(bio, sep, enc_pref: Optional[str], chunksize=None, nrows=None, engine_hint=None):
    enc_try = []
    if enc_pref and enc_pref != "auto": enc_try.append(enc_pref)
    enc_try += ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in enc_try + ["latin1//replace"]:
        try:
            bio.seek(0)
            kwargs = dict(sep=sep, on_bad_lines="skip")
            if chunksize is not None: kwargs["chunksize"] = chunksize
            if nrows is not None: kwargs["nrows"] = nrows
            if engine_hint: kwargs["engine"] = engine_hint
            else:
                kwargs["engine"] = "python" if sep is None else "c"
                if kwargs["engine"] == "c": kwargs["low_memory"] = False
            if enc.endswith("//replace"):
                kwargs["encoding"] = enc.split("//")[0]; kwargs["encoding_errors"] = "replace"
            else:
                kwargs["encoding"] = enc
            return pd.read_csv(bio, **kwargs)
        except Exception as e:
            last_err = e; continue
    raise last_err if last_err else RuntimeError("CSV read failed")

# -------------------------
# Streamed profiling
# -------------------------
def streamed_profile_csv(bio: io.BytesIO, encoding: Optional[str], sep: Optional[str],
                         preview_rows:int=PREVIEW_ROWS):
    enc_chain = []
    if encoding and encoding != "auto": enc_chain.append(encoding)
    enc_chain += ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    last_err = None
    for enc_try in enc_chain + ["latin1//replace"]:
        try:
            bio.seek(0)
            engine = "python" if sep is None else "c"
            read_kwargs = dict(sep=sep, on_bad_lines="skip", chunksize=CHUNK_SIZE_PROFILE, engine=engine)
            if enc_try.endswith("//replace"):
                read_kwargs["encoding_errors"] = "replace"; read_kwargs["encoding"] = enc_try.split("//")[0]
            else:
                read_kwargs["encoding"] = enc_try

            reader = pd.read_csv(bio, **read_kwargs)

            n_rows, cols = 0, None
            dtypes_guess, missing_sums, uniq_samples = {}, {}, {}
            head_buf = []

            while True:
                try:
                    chunk = next(reader)
                except StopIteration:
                    break
                except Exception as ue:
                    last_err = ue; raise ue

                chunk = _clean_columns(chunk)
                if cols is None: cols = list(chunk.columns)
                n_rows += len(chunk)

                if len(head_buf) < preview_rows:
                    take = min(preview_rows - len(head_buf), len(chunk))
                    head_buf.append(chunk.head(take))

                for c in chunk.columns:
                    missing_sums[c] = missing_sums.get(c, 0) + chunk[c].isna().sum()
                    uset = uniq_samples.setdefault(c, set())
                    with contextlib.suppress(Exception):
                        for v in chunk[c].dropna().astype(str).head(50).values:
                            if len(uset) < 500: uset.add(v)

                if not dtypes_guess:
                    dtypes_guess = {c: str(t) for c, t in chunk.dtypes.items()}

            if n_rows == 0:
                return {"rows":0,"cols":0,"missing_pct_overall":0.0,"dtypes":{}, "columns":[]}, pd.DataFrame()

            head_df = pd.concat(head_buf, ignore_index=True) if head_buf else pd.DataFrame(columns=cols)
            head_df = _coerce_light(head_df)

            col_meta, id_like = [], []
            for c in cols:
                miss_pct = float(missing_sums.get(c,0)) / max(1,n_rows) * 100.0
                uniq_pct_approx = (len(uniq_samples.get(c,set())) / max(1,n_rows)) * 100.0
                if uniq_pct_approx >= 90.0: id_like.append(c)
                exvals = list(list(uniq_samples.get(c,set()))[:3])
                col_meta.append({
                    "column": c,
                    "dtype": dtypes_guess.get(c,"unknown"),
                    "missing_pct": round(miss_pct,3),
                    "unique_pct": round(uniq_pct_approx,3),
                    "examples": exvals
                })
            missing_overall = float(np.mean([m["missing_pct"] for m in col_meta])) if col_meta else 0.0

            # candidate target guess
            candidate_target = None
            name_hits = [c for c in cols if str(c).lower() in {"target","label","y","class","status","price","amount","value"}]
            if name_hits:
                candidate_target = name_hits[0]
            else:
                low_card = [(c, len(uniq_samples.get(c,set()))) for c in cols if len(uniq_samples.get(c,set()))<=50 and c not in id_like]
                if low_card:
                    candidate_target = sorted(low_card, key=lambda x: x[1])[0][0]

            prof = {
                "rows": int(n_rows),
                "cols": int(len(cols)),
                "missing_pct_overall": round(missing_overall,4),
                "dtypes": dtypes_guess,
                "id_like_columns": id_like,
                "candidate_target": candidate_target,
                "columns": col_meta,
                "_encoding_used": enc_try if not enc_try.endswith("//replace") else enc_try.replace("//", " (errors=") + ")"
            }
            return prof, head_df.head(PREVIEW_ROWS)

        except Exception:
            continue

    raise last_err if last_err else RuntimeError("CSV streamed profiling failed.")

def detect_task_from_profile(prof: dict) -> str:
    tgt = prof.get("candidate_target")
    # Heuristic: if target exists and looks numeric → regression; else classification; else anomaly
    if tgt:
        # peek at dtype guess
        dt = str(prof.get("dtypes", {}).get(tgt, "")).lower()
        if any(k in dt for k in ["float", "int", "number"]): return "regression"
        return "classification"
    return "anomaly"

def suggest_strategy(rows:int, cols:int, baseline_ok: Optional[bool]=None)->dict:
    choice, reason = "single", "ULTRA-LITE default (memory guard)"
    if baseline_ok is False: choice, reason = "ensemble", "baseline under target metric"
    return {"strategy": choice, "reason": reason}

# -------------------------
# Session keys (tiny)
# -------------------------
for k in ["data_sha1","data_name","data_path","meta_path","profile","preview","last_run_dir"]:
    st.session_state.setdefault(k, None)

# -------------------------
# Synthetic demo helpers (60-sec showcase)
# -------------------------
def _demo_make_df(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0,1,size=n).astype(np.float32)
    X2 = rng.normal(0,1,size=n).astype(np.float32)
    cat = rng.choice(["A","B","C","D"], size=n, p=[0.4,0.3,0.2,0.1])
    # add currency-like noise to show coercer working
    y = (X1 + 0.5*X2 + (cat == "A").astype(int) + rng.normal(0,0.5,size=n).astype(np.float32)) * 1000
    y = np.where(rng.random(size=n) < 0.02, None, y)  # a few NAs
    y = np.round(y,2)
    df = pd.DataFrame({"x1":X1, "x2":X2, "cat":cat, "funding_total_usd":[f"${v:,.2f}" if v is not None else "-" for v in y]})
    return df

def _demo_stage_into_session(df: pd.DataFrame):
    raw_buf = io.BytesIO()
    df.to_csv(raw_buf, index=False)
    raw = raw_buf.getvalue()
    sha1 = hashlib.sha1(raw).hexdigest()
    raw_path = DATA_DIR / f"{sha1}.bin"
    with open(raw_path, "wb") as f: f.write(raw)
    prof, preview = streamed_profile_csv(io.BytesIO(raw), "utf-8", ",", PREVIEW_ROWS)
    meta = {
        "filename": "demo.csv",
        "sha1": sha1,
        "size_human": _bytes_h(len(raw)),
        "io": {"kind":"csv","sheet":None,"encoding":"utf-8","sep":",","size":len(raw)},
        "profile": prof,
        "created_at": time.time()
    }
    meta_path = DATA_DIR / f"{sha1}_meta.json"
    save_json(meta, meta_path)
    st.session_state.data_sha1 = sha1
    st.session_state.data_name = "demo.csv"
    st.session_state.data_path = str(raw_path)
    st.session_state.meta_path = str(meta_path)
    st.session_state.profile = prof
    st.session_state.preview = preview.head(PREVIEW_ROWS).copy()

def _demo_autorun_full(seed=7):
    """One-click 60s demo: stage -> train -> SLO -> gate."""
    df = _demo_make_df(n=2000, seed=seed)
    _demo_stage_into_session(df)

    meta = json.loads(Path(st.session_state.meta_path).read_text())
    prof = meta.get("profile", {})
    target = "funding_total_usd"
    task = "regression"
    strategy = "single"

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.pipeline import Pipeline

    Xy_df, y = load_for_train(st.session_state.data_path, meta.get("io",{}), target, cap=min(2000, ROW_CAP_TRAIN))
    feats = [c for c in Xy_df.columns if c != target]
    if len(feats) == 0:
        raise RuntimeError("No features available in demo dataset.")
    X = Xy_df[feats]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_list, mk_preproc = get_models(task, strategy)
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feats if (not pd.api.types.is_numeric_dtype(X[c])) and (not pd.api.types.is_datetime64_any_dtype(X[c]))]
    preproc = mk_preproc(num_cols, cat_cols)

    best_name, best_model, best_metrics = None, None, None
    for name, core in model_list:
        pipe = Pipeline([("prep", preproc), ("model", core)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = {"r2": float(r2_score(y_test, y_pred)),
                   "mae": float(mean_absolute_error(y_test, y_pred)),
                   "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))}
        key = (metrics.get("r2",-1.0), -metrics.get("rmse", 1e9))
        if best_metrics is None or key > (best_metrics.get("r2",-1.0), -best_metrics.get("rmse",1e9)):
            best_name, best_model, best_metrics = name, pipe, metrics

    run_id = f"demo_{st.session_state.data_sha1[:8]}_{int(time.time())}"
    run_dir = MODEL_DIR / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id, "data_sha1": st.session_state.data_sha1, "data_name": st.session_state.data_name,
        "task": task, "strategy": strategy, "target": target, "best_model": best_name,
        "metrics": best_metrics, "rows_sampled": int(len(X)), "created_at": pd.Timestamp.utcnow().isoformat()+"Z",
        "_demo": True
    }
    try:
        import joblib
        joblib.dump(best_model, run_dir / "model.pkl")
    except Exception:
        pass
    save_json(report, run_dir / "report.json")

    slo = {"task": task, "timestamp": pd.Timestamp.utcnow().isoformat()+"Z",
           "metrics": best_metrics, "dataset_sha1": st.session_state.data_sha1, "run_dir": str(run_dir)}
    _write_gateway_file("gateway_slo_eval.json", slo)

    passed = slo.get("metrics",{}).get("r2", 0.0) >= 0.5
    gate = {"run_dir": str(run_dir), "ts": pd.Timestamp.utcnow().isoformat()+"Z",
            "slo_pass": passed, "decision": "promote" if passed else "hold",
            "reason": "SLO thresholds met" if passed else "Thresholds not met"}
    _write_gateway_file("gateway_promotion_gate.json", gate)

    st.session_state.last_run_dir = str(run_dir)
    return {"report": report, "slo": slo, "gate": gate}

# -------------------------
# UI: Home
# -------------------------
def tab_home():
    st.header("🏠 Home")
    st.write("**ULTRA-LITE** build: streamed profiling, tiny models, dense-safe preprocessing. "
             "Models sleep until you press **Train** or run the **Auto Demo**. "
             "Ensembles/XGB/SHAP minimized for memory.")
    with st.expander("🎬 60-Second Demo"):
        st.write("Option A: stage demo data only. Option B: full end-to-end auto-run.")
        c1, c2 = st.columns(2)
        if c1.button("Stage Demo Data (synthetic 2k rows)", use_container_width=True):
            df = _demo_make_df()
            _demo_stage_into_session(df)
            st.success("Demo data staged. Go to **Train** → **Train now** → then **Evaluate/Promote**.")
        if c2.button("Run FULL Auto Demo (≈60s)", use_container_width=True):
            with st.spinner("Staging → Training → Evaluating → Promoting…"):
                out = _demo_autorun_full()
            st.success("Demo complete. Peek the Evaluate/Promote/Monitor tabs.")
            st.json(out["report"])

# -------------------------
# UI: Data (streamed ingest)
# -------------------------
def tab_data():
    st.header("📂 Data")
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx","xls"])
    if not uploaded:
        st.info("Upload a file to begin."); return

    raw = uploaded.read(); uploaded.seek(0)
    sha1 = hashlib.sha1(raw).hexdigest()
    size_h = _bytes_h(len(raw))
    fname = uploaded.name

    # store raw file once (single active dataset per session)
    for p in DATA_DIR.glob("*"):
        try:
            p.unlink() if p.is_file() else shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    raw_path = DATA_DIR / f"{sha1}.bin"
    with open(raw_path, "wb") as f: f.write(raw)

    is_excel = fname.lower().endswith((".xlsx",".xls")) or raw[:4] == b"PK\x03\x04"
    sheet_name = None

    if is_excel:
        try:
            xl = pd.ExcelFile(io.BytesIO(raw))
            sheets = xl.sheet_names
            sheet_name = st.selectbox("Excel sheet", sheets, index=0)
            df_prev = xl.parse(sheet_name, nrows=PREVIEW_ROWS)
            df_prev = _coerce_light(_clean_columns(df_prev))
            prof = {
                "rows": None, "cols": int(df_prev.shape[1]),
                "missing_pct_overall": float(df_prev.isna().mean().mean()*100.0),
                "dtypes": {c:str(t) for c,t in df_prev.dtypes.items()},
                "id_like_columns": [],
                "candidate_target": next((c for c in df_prev.columns if str(c).lower() in {"target","label","y","class","status","price","amount","value"}), None),
                "columns": [{"column":c,"dtype":str(df_prev[c].dtype),
                             "missing_pct": float(df_prev[c].isna().mean()*100.0),
                             "unique_pct": None,
                             "examples": list(df_prev[c].dropna().astype(str).head(3).values)} for c in df_prev.columns]
            }
            task = detect_task_from_profile(prof)
            st.success("✅ File staged (Excel). Using preview-based profile.")
            preview = df_prev
            io_meta = {"kind":"excel","sheet":sheet_name,"encoding":None,"sep":None,"size":len(raw)}
        except Exception as e:
            st.error(f"Failed to read Excel preview: {e}"); return
    else:
        enc_choice = st.selectbox("Encoding", ["auto","utf-8","utf-8-sig","cp1252","latin1","latin1 (force replace)"], index=0)
        sep_choice = st.selectbox("Delimiter", ["auto", ",", ";", "\\t", "|"], index=0)
        sep_val = {"auto":None,",":",",";":";","\\t":"\t","|":"|"}[sep_choice]
        enc_val = None if enc_choice == "auto" else ("latin1//replace" if "force replace" in enc_choice else enc_choice)
        prof, preview = streamed_profile_csv(io.BytesIO(raw), enc_val or "auto", sep_val, PREVIEW_ROWS)
        task = detect_task_from_profile(prof)
        io_meta = {"kind":"csv","sheet":None,"encoding":enc_val or "auto","sep":sep_choice,"size":len(raw)}
        st.success("✅ File staged and profiled (streamed).")

    if (isinstance(prof, dict) and (prof.get("cols", 0) == 0)) or (preview is None):
        st.error("Could not profile this file. Try adjusting delimiter/encoding."); return

    meta = {
        "filename": fname,
        "sha1": sha1,
        "size_human": size_h,
        "io": io_meta,
        "profile": prof,
        "created_at": time.time()
    }
    meta_path = DATA_DIR / f"{sha1}_meta.json"
    save_json(meta, meta_path)

    st.session_state.data_sha1 = sha1
    st.session_state.data_name = fname
    st.session_state.data_path = str(raw_path)
    st.session_state.meta_path = str(meta_path)
    st.session_state.profile = prof
    st.session_state.preview = preview.head(PREVIEW_ROWS).copy()

    c1,c2,c3,c4 = st.columns([1,1,1,1.2])
    c1.metric("Approx Rows", f"{prof.get('rows','?')}")
    c2.metric("Cols", f"{prof.get('cols','?')}")
    c3.metric("Missing %", f"{(prof.get('missing_pct_overall') or 0):.2f}%")
    c4.metric("Size", size_h)
    st.caption(f"Suggested task: **{task}**")
    st.caption(f"Strategy hint: **single** — ULTRA-LITE default")

    with st.expander("Preview (first rows)"):
        st.dataframe(preview, use_container_width=True, height=360)

# -------------------------
# Lazy model factory (imports on demand, dense-safe + tiny)
# -------------------------
def get_models(task:str, strategy:str):
    # Lazy imports so models sleep until used
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.impute import SimpleImputer

    # ---- OHE compatibility shim (sklearn >=1.4 uses sparse_output)
    def _make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    _ohe = _make_ohe()

    def preproc(num_cols, cat_cols):
        # Numeric cleaner → impute → scale; Categoricals → impute → OHE; cast to float32 at the end
        def _numeric_clean_block(Xdf: pd.DataFrame):
            Xdf = Xdf.copy()
            for c in num_cols:
                if c in Xdf.columns:
                    Xdf[c] = Xdf[c].map(_to_float_smart)
            return Xdf[num_cols].values

        def _categorical_block(Xdf: pd.DataFrame):
            return Xdf[cat_cols].astype("object").values

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer

        num_pipe = Pipeline([
            ("clean", FunctionTransformer(lambda X: np.column_stack([pd.Series(X[:,i]).map(_to_float_smart)
                                                for i in range(X.shape[1])]) if X.size else X, validate=False)),
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True))
        ])

        # We need ColumnTransformer that accepts DataFrame; to keep memory tiny we use functions directly on DF
        return Pipeline([
            ("ct", ColumnTransformer(
                transformers=[
                    ("num_df", Pipeline([
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler(with_mean=True))
                    ]),
                     num_cols),
                    ("cat_df", Pipeline([
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", _ohe)
                    ]),
                     cat_cols),
                ],
                remainder="drop",
                sparse_threshold=0.0  # force dense
            )),
            ("cast32", FunctionTransformer(lambda X: X.astype(np.float32), accept_sparse=True))
        ])

    # Tiny models (quantized by depth/estimators)
    if task == "classification":
        if strategy == "single" or not ENABLE_ENSEMBLES:
            return [
                ("logreg", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)),
                ("rf", RandomForestClassifier(
                    n_estimators=32, max_depth=7, min_samples_leaf=4, n_jobs=-1, class_weight="balanced", random_state=42
                )),
            ], preproc
        else:
            from sklearn.ensemble import VotingClassifier
            base = [
                ("logreg", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)),
                ("rf", RandomForestClassifier(n_estimators=24, max_depth=6, min_samples_leaf=4, n_jobs=-1, class_weight="balanced", random_state=42)),
            ]
            return [("softvote", VotingClassifier(estimators=base, voting="soft"))], preproc

    if task == "regression":
        if strategy == "single" or not ENABLE_ENSEMBLES:
            return [
                ("lin", LinearRegression()),
                ("rf", RandomForestRegressor(n_estimators=32, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=42)),
            ], preproc
        else:
            from sklearn.ensemble import VotingRegressor
            base = [
                ("lin", LinearRegression()),
                ("rf", RandomForestRegressor(n_estimators=24, max_depth=7, min_samples_leaf=4, n_jobs=-1, random_state=42)),
            ]
            return [("vote", VotingRegressor(estimators=base))], preproc

    if task == "anomaly":
        from sklearn.ensemble import IsolationForest
        # IF ignores target; preproc will only transform features
        return [("iforest", IsolationForest(
            n_estimators=80, max_samples="auto", max_features=1.0, contamination="auto", random_state=42,
        ))], preproc

    return [], preproc

# -------------------------
# Safe loader for training (sampled, engine-safe, with robust target coercion)
# -------------------------
def load_for_train(raw_path: str, io_meta: dict, target: Optional[str], cap:int=ROW_CAP_TRAIN) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    raw = Path(raw_path).read_bytes()
    is_excel = raw[:4] == b"PK\x03\x04" or (st.session_state.data_name or "").lower().endswith((".xlsx",".xls"))
    if is_excel:
        xl = pd.ExcelFile(io.BytesIO(raw))
        sheet = io_meta.get("sheet") or 0
        df = xl.parse(sheet, nrows=cap)
    else:
        enc = io_meta.get("encoding", "auto")
        sep_choice = io_meta.get("sep", None)
        sep = None if sep_choice in (None, "auto") else {"\\t":"\t"}.get(sep_choice, sep_choice)
        engine = "python" if sep is None else "c"
        df = _read_csv_try_encodings(
            bio=io.BytesIO(raw),
            sep=sep,
            enc_pref=enc,
            nrows=cap,
            engine_hint=engine
        )
    df = _coerce_light(_clean_columns(df))
    y = None
    if target:
        if target not in df.columns:
            return df, None
        y_raw = df[target]
        # Robust coercion for regression-like targets; keep as-is for classification if non-numeric
        y_num = y_raw.map(_to_float_smart)
        # Decide: if >60% numeric after coercion → treat as numeric target
        frac_num = float(pd.notna(y_num).mean())
        if frac_num >= 0.60:
            y = y_num
        else:
            y = y_raw  # likely categorical target
        mask = ~y.isna() if pd.api.types.is_numeric_dtype(y) else y.notna()
        df, y = df.loc[mask], y.loc[mask]
    return df, y

# -------------------------
# Training
# -------------------------
def tab_train():
    st.header("🛠️ Train")
    if not st.session_state.data_path or not st.session_state.meta_path:
        st.info("Upload data in **Data** tab first or run the **Demo**."); return
    meta_path = Path(st.session_state.meta_path)
    if not meta_path.exists():
        st.warning("Data wasn’t profiled successfully. Go back to **Data** and re-upload.")
        return
    meta = json.loads(meta_path.read_text())
    prof = meta.get("profile", {})
    df_preview = st.session_state.preview

    tgt_default = prof.get("candidate_target")
    opts = [None] + (list(df_preview.columns) if df_preview is not None else [])
    idx = 0
    if df_preview is not None and tgt_default in df_preview.columns:
        idx = 1 + list(df_preview.columns).index(tgt_default)
    c1,c2,c3 = st.columns([1,1,1])
    target = c1.selectbox("Target (optional for anomaly)", opts, index=idx)
    auto_task = detect_task_from_profile(prof)
    task = c2.selectbox("Task", ["classification","regression","anomaly"],
                        index={"classification":0,"regression":1,"anomaly":2}[auto_task])
    strategy = c3.selectbox("Strategy", ["single","ensemble"], index=0)
    st.caption("ULTRA-LITE: training rows capped for stability. Models are tiny & dense-cast to float32.")

    if task in ("classification","regression") and target is None:
        st.warning("Pick a target or switch to anomaly."); return

    run_btn = st.button("Train now", type="primary", use_container_width=True)
    if not run_btn: return

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
    from sklearn.pipeline import Pipeline

    Xy_df, y = load_for_train(st.session_state.data_path, meta.get("io",{}), target, cap=ROW_CAP_TRAIN)
    feats = [c for c in Xy_df.columns if c != target] if target else list(Xy_df.columns)
    if len(feats) == 0:
        st.error("No usable feature columns after cleaning."); return

    X = Xy_df[feats]
    stratify = None
    if task=="classification" and target is not None and y is not None:
        vc = pd.Series(y).value_counts(dropna=True)
        st.caption(f"Class counts (sampled): { {str(k):int(v) for k,v in vc.head(20).items()} }{' …' if len(vc)>20 else ''}")
        if pd.Series(y).nunique(dropna=True)>1 and int(vc.min())>=2: stratify=y
    if task in ("classification","regression"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    model_list, mk_preproc = get_models(task, strategy)
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(X[c])]
    # include object columns as categorical (datetimes excluded)
    cat_cols = [c for c in feats if (not pd.api.types.is_numeric_dtype(X[c])) and (not pd.api.types.is_datetime64_any_dtype(X[c]))]
    preproc = mk_preproc(num_cols, cat_cols)

    progress = st.progress(0.0, text="Training…")
    best_name, best_model, best_metrics = None, None, None

    def pk(task, m:Dict[str,Any])->float:
        if task=="classification": return m.get("auc", m.get("f1", -1e9))
        if task=="regression": return m.get("r2", -1e9)
        if task=="anomaly": return m.get("score_mean", -1e9)
        return -1e9

    for j,(name, core) in enumerate(model_list, start=1):
        progress.progress(j/len(model_list), text=f"Training {name}…")
        try:
            if task in ("classification","regression"):
                pipe = Pipeline([("prep", preproc), ("model", core)])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                if task=="classification":
                    metrics = {"accuracy": float(accuracy_score(y_test, y_pred)),
                               "f1": float(f1_score(y_test, y_pred, average="weighted"))}
                    with contextlib.suppress(Exception):
                        if hasattr(pipe.named_steps["model"], "predict_proba") and pd.Series(y_test).nunique()==2:
                            proba = pipe.predict_proba(X_test)
                            metrics["auc"] = float(roc_auc_score(y_test, proba[:,1]))
                else:
                    metrics = {"r2": float(r2_score(y_test, y_pred)),
                               "mae": float(mean_absolute_error(y_test, y_pred)),
                               "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))}
            else:
                from sklearn.pipeline import Pipeline as _P
                pipe = _P([("prep", preproc), ("model", core)])
                pipe.fit(X_train)
                # IsolationForest: higher anomaly → lower score_samples; flip sign
                raw_scores = pipe.named_steps["model"].score_samples(pipe.named_steps["prep"].transform(X_test))
                scores = (-np.asarray(raw_scores).ravel())
                metrics = {"score_mean": float(np.mean(scores)), "score_std": float(np.std(scores))}
        except Exception as e:
            st.warning(f"{name} failed: {e}"); continue

        if (best_metrics is None) or pk(task, metrics) > pk(task, best_metrics):
            best_name, best_model, best_metrics = name, pipe, metrics

    progress.empty()
    if best_model is None:
        st.error("No model trained successfully."); return

    run_id = f"run_{st.session_state.data_sha1[:8]}_{int(time.time())}"
    run_dir = MODEL_DIR / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id, "data_sha1": st.session_state.data_sha1, "data_name": st.session_state.data_name,
        "task": task, "strategy": strategy, "target": target, "best_model": best_name,
        "metrics": best_metrics, "rows_sampled": int(len(X)), "created_at": pd.Timestamp.utcnow().isoformat()+"Z"
    }
    try:
        import joblib
        joblib.dump(best_model, run_dir / "model.pkl")
    except Exception as e:
        st.caption(f"Model save skipped (joblib not available?): {e}")
    save_json(report, run_dir / "report.json")

    slo = {"task": task, "timestamp": pd.Timestamp.utcnow().isoformat()+"Z",
           "metrics": best_metrics, "dataset_sha1": st.session_state.data_sha1, "run_dir": str(run_dir)}
    _write_gateway_file("gateway_slo_eval.json", slo)

    st.session_state.last_run_dir = str(run_dir)
    if task=="regression":
        st.success(f"✅ Trained **{best_name}** on {min(ROW_CAP_TRAIN, prof.get('rows') or ROW_CAP_TRAIN):,} rows (cap). R²={best_metrics.get('r2',0):.3f}")
    elif task=="classification":
        st.success(f"✅ Trained **{best_name}**. F1={best_metrics.get('f1',0):.3f} AUC={best_metrics.get('auc',float('nan'))}")
    else:
        st.success(f"✅ Trained **{best_name}**. score_mean={best_metrics.get('score_mean',0):.3f}")
    st.json(best_metrics)

    del X, Xy_df, y
    gc.collect()

# -------------------------
# Evaluate
# -------------------------
def tab_evaluate():
    st.header("📈 Evaluate")
    if not st.session_state.last_run_dir:
        st.info("Train a model first."); return
    rep_p = Path(st.session_state.last_run_dir) / "report.json"
    if rep_p.exists():
        st.json(json.loads(rep_p.read_text()))
    else:
        st.info("No report found.")

    slo_p = SESSION_DIR / "gateway_slo_eval.json"
    if slo_p.exists():
        st.subheader("SLO Snapshot"); st.json(json.loads(slo_p.read_text()))
    else:
        st.info("No SLO file yet.")

# -------------------------
# Promote
# -------------------------
def tab_promote():
    st.header("🚀 Promote")
    slo_p = SESSION_DIR / "gateway_slo_eval.json"
    slo = json.loads(slo_p.read_text()) if slo_p.exists() else {}
    st.subheader("SLO Check"); st.json(slo or {"note":"No SLO"})
    def meets_slo(task, m):
        if task=="classification": return (m.get("auc") or m.get("f1",0.0)) >= 0.75
        if task=="regression": return m.get("r2", -1.0) >= 0.5
        if task=="anomaly": return m.get("score_mean", 0.0) > 0.0  # placeholder
        return False
    if st.button("Run Promotion Gate", type="primary", use_container_width=True):
        passed = bool(slo) and meets_slo(slo.get("task",""), slo.get("metrics",{}))
        gate = {"run_dir": st.session_state.last_run_dir, "ts": pd.Timestamp.utcnow().isoformat()+"Z",
                "slo_pass": passed, "decision": "promote" if passed else "hold",
                "reason": "SLO thresholds met" if passed else "Thresholds not met / no SLO"}
        _write_gateway_file("gateway_promotion_gate.json", gate)
        st.success(f"Decision: **{gate['decision']}** (written to session & ART_DIR)")

# -------------------------
# Monitor
# -------------------------
def tab_monitor():
    st.header("📡 Monitor")
    slo_p = SESSION_DIR / "gateway_slo_eval.json"
    if slo_p.exists(): st.json(json.loads(slo_p.read_text()))
    else: st.info("No SLO yet.")

# -------------------------
# App
# -------------------------
tabs = st.tabs(["Home","Data","Train","Evaluate","Promote","Monitor"])
with tabs[0]: tab_home()
with tabs[1]:
    try: tab_data()
    except Exception as e: st.error("Data tab error:"); st.exception(e)
with tabs[2]: tab_train()
with tabs[3]: tab_evaluate()
with tabs[4]: tab_promote()
with tabs[5]: tab_monitor()
