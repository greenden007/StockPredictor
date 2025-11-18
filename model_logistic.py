# model_compare.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from typing import Tuple, Dict

CSV_PATH = "processed_data/AAPL.csv"

def read_prices(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find file: {csv_path}")
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = (df.columns.astype(str).str.strip().str.lower().str.replace(r"\s+","_",regex=True))
    # find date
    date_col = next((c for c in ["date","datetime","time","timestamp"] if c in df.columns), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"}).sort_values("date").reset_index(drop=True)
    # choose prices
    close_c = "adj_close" if "adj_close" in df.columns else "close"
    if close_c not in df.columns or "volume" not in df.columns:
        raise ValueError("Need close/adj_close and volume columns")
    for c in [close_c, "volume"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","",regex=False).str.replace("$","",regex=False), errors="coerce")
    # optional highs/lows/opens if present
    for c in ["open","high","low"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","",regex=False).str.replace("$","",regex=False), errors="coerce")
    return df.rename(columns={close_c:"close"}).dropna(subset=["close","volume"]).reset_index(drop=True)

# Feature engineering: technical indicators
def ta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["close"].pct_change()

    # Moving averages & crossover
    out["sma5"] = out["close"].rolling(5).mean()
    out["sma10"] = out["close"].rolling(10).mean()
    out["sma20"] = out["close"].rolling(20).mean()
    out["sma5_over_10"] = (out["sma5"] / out["sma10"]) - 1

    # RSI Calculation (Relative Strength Index) with 14-day window
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    out["rsi14"] = 100 - (100 / (1 + rs))

    # MACD (Trend/Momentum Indicator)
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Volatility & volume dynamics
    out["vol10"] = out["ret1"].rolling(10).std()
    out["v_chg"] = out["volume"].pct_change()

    # Intraday range & gaps
    if {"high","low"}.issubset(out.columns):
        out["range_pct"] = (out["high"] - out["low"]) / out["close"].shift(1)
    else:
        out["range_pct"] = np.nan
    if "open" in out.columns:
        out["gap_open"] = (out["open"] - out["close"].shift(1)) / out["close"].shift(1)
    else:
        out["gap_open"] = np.nan

    # Target: next-day direction
    out["target"] = (out["close"].pct_change().shift(-1) > 0).astype(int)

    # Drop rows with NaNs from rolling windows
    out = out.dropna().reset_index(drop=True)
    return out

# Time series CV to choose probability threshold
def cv_choose_threshold(pipe, X, y, n_splits=5) -> float:
    ts = TimeSeriesSplit(n_splits=n_splits)
    best_thr, best_score = 0.5, -1.0
    for thr in np.linspace(0.2, 0.8, 25):
        scores = []
        for tr, va in ts.split(X):
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[va])[:, 1]
            pred = (p >= thr).astype(int)
            scores.append(balanced_accuracy_score(y[va], pred))
        m = float(np.mean(scores))
        if m > best_score:
            best_score, best_thr = m, thr
    return best_thr

# Train and Test Evaluation
def final_holdout_eval(pipe, X, y, test_frac=0.2, threshold: float = 0.5) -> Dict[str, float]:
    cut = int(len(X) * (1 - test_frac))
    pipe.fit(X[:cut], y[:cut])
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(X[cut:])[:, 1]
        yhat = (p >= threshold).astype(int)
    else:
        yhat = pipe.predict(X[cut:])
    acc = accuracy_score(y[cut:], yhat)
    bacc = balanced_accuracy_score(y[cut:], yhat)
    f1 = f1_score(y[cut:], yhat)
    print("Confusion Matrix:\n", confusion_matrix(y[cut:], yhat))
    print(classification_report(y[cut:], yhat, digits=3))
    return {"accuracy": acc, "balanced_acc": bacc, "f1": f1}


df = read_prices(CSV_PATH)
df = ta_features(df)

feature_cols = [
    "ret1","sma5","sma10","sma20","sma5_over_10",
    "rsi14","macd","macd_signal","macd_hist",
    "vol10","v_chg","range_pct","gap_open"
]
X = df[feature_cols].values
y = df["target"].values

# Models to compare
logreg = Pipeline([
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=400, class_weight="balanced"))
])
rf = RandomForestClassifier(
    n_estimators=400, max_depth=5, min_samples_leaf=5,
    class_weight="balanced_subsample", random_state=42
)

# Time-series CV threshold tuning for LogReg
thr = cv_choose_threshold(logreg, X, y, n_splits=5)
print(f"Chosen probability threshold (via TS-CV): {thr:.3f}")

print("\n=== Logistic Regression (final holdout) ===")
lr_metrics = final_holdout_eval(logreg, X, y, test_frac=0.2, threshold=thr)
print({k: round(v, 3) for k, v in lr_metrics.items()})

print("\n=== Random Forest (final holdout) ===")
# RF outputs classes; use its probabilities + same threshold for consistency
rf_pipe = rf
# Evaluate RF with default 0.5 cut (and print too with tuned thr)
cut = int(len(X) * 0.8)
rf_pipe.fit(X[:cut], y[:cut])
proba = rf_pipe.predict_proba(X[cut:])[:, 1]
rf_pred = (proba >= thr).astype(int)  # use tuned threshold from CV for fairness
acc = accuracy_score(y[cut:], rf_pred)
bacc = balanced_accuracy_score(y[cut:], rf_pred)
f1  = f1_score(y[cut:], rf_pred)
print("Confusion Matrix:\n", confusion_matrix(y[cut:], rf_pred))
print(classification_report(y[cut:], rf_pred, digits=3))
print({"accuracy": round(acc,3), "balanced_acc": round(bacc,3), "f1": round(f1,3)})
