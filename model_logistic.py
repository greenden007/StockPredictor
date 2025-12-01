# model_compare.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from typing import Dict

# ===== VISUALIZATION LIBRARIES =====
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "processed_data/AAPL.csv"


def read_prices(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find file: {csv_path}")
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True))

    date_col = next((c for c in ["date", "datetime", "time", "timestamp"] if c in df.columns),
                    df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"}).sort_values("date")

    close_c = "adj_close" if "adj_close" in df.columns else "close"
    if close_c not in df.columns or "volume" not in df.columns:
        raise ValueError("Need close/adj_close and volume columns")

    for c in [close_c, "volume"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("$", ""),
                              errors="coerce")

    for c in ["open", "high", "low"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("$", ""),
                                  errors="coerce")

    return df.rename(columns={close_c: "close"}).dropna(subset=["close", "volume"]).reset_index(drop=True)


# ============================================================
#                     TECHNICAL INDICATORS
# ============================================================
def ta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret1"] = out["close"].pct_change()
    out["sma5"] = out["close"].rolling(5).mean()
    out["sma10"] = out["close"].rolling(10).mean()
    out["sma20"] = out["close"].rolling(20).mean()
    out["sma5_over_10"] = (out["sma5"] / out["sma10"]) - 1

    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["vol10"] = out["ret1"].rolling(10).std()
    out["v_chg"] = out["volume"].pct_change()

    if {"high", "low"}.issubset(out.columns):
        out["range_pct"] = (out["high"] - out["low"]) / out["close"].shift(1)
    else:
        out["range_pct"] = np.nan

    if "open" in out.columns:
        out["gap_open"] = (out["open"] - out["close"].shift(1)) / out["close"].shift(1)
    else:
        out["gap_open"] = np.nan

    out["target"] = (out["close"].pct_change().shift(-1) > 0).astype(int)

    return out.dropna().reset_index(drop=True)


# ============================================================
#               TIME SERIES CROSS-VALIDATION
# ============================================================
def cv_choose_threshold(pipe, X, y, n_splits=5) -> float:
    ts = TimeSeriesSplit(n_splits=n_splits)
    best_thr, best_score = 0.5, -1

    for thr in np.linspace(0.2, 0.8, 25):
        scores = []
        for tr, va in ts.split(X):
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[va])[:, 1]
            pred = (p >= thr).astype(int)
            scores.append(balanced_accuracy_score(y[va], pred))

        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_thr = thr

    return best_thr


# ============================================================
#                     FINAL HOLDOUT EVALUATION
# ============================================================
def final_holdout_eval(pipe, X, y, test_frac=0.2, threshold=0.5):
    cut = int(len(X) * (1 - test_frac))
    pipe.fit(X[:cut], y[:cut])

    p = pipe.predict_proba(X[cut:])[:, 1]
    yhat = (p >= threshold).astype(int)

    cm = confusion_matrix(y[cut:], yhat)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y[cut:], yhat))

    return (cm, y[cut:], yhat)


# ============================================================
#                      VISUALIZATION UTILS
# ============================================================
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Down", "Pred Up"],
                yticklabels=["True Down", "True Up"])
    plt.title(title)
    plt.show()


def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[idx], importances[idx])
    plt.title("Random Forest Feature Importances")
    plt.show()


def plot_predictions(df, y_true, y_pred, cut):
    plt.figure(figsize=(12, 3))
    plt.plot(df["date"][cut:], y_true, label="Actual", alpha=0.7)
    plt.plot(df["date"][cut:], y_pred, label="Predicted", alpha=0.7)
    plt.title("Predicted vs Actual Up/Down Direction")
    plt.legend()
    plt.show()


def plot_price_with_sma(df, limit=200):
    d = df.tail(limit)
    plt.figure(figsize=(12, 4))
    plt.plot(d["date"], d["close"], label="Close")
    plt.plot(d["date"], d["sma5"], label="SMA5")
    plt.plot(d["date"], d["sma20"], label="SMA20")
    plt.title("Price with Moving Averages")
    plt.legend()
    plt.show()


def plot_rsi(df, limit=200):
    d = df.tail(limit)
    plt.figure(figsize=(12, 3))
    plt.plot(d["date"], d["rsi14"], label="RSI(14)")
    plt.axhline(70, color="red", linestyle="--")
    plt.axhline(30, color="green", linestyle="--")
    plt.title("RSI(14)")
    plt.legend()
    plt.show()


def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def plot_precision_recall(y_true, y_scores, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


# ============================================================
#                     MAIN EXECUTION
# ============================================================
df = read_prices(CSV_PATH)
df = ta_features(df)

feature_cols = [
    "ret1", "sma5", "sma10", "sma20", "sma5_over_10",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "vol10", "v_chg", "range_pct", "gap_open"
]

X = df[feature_cols].values
y = df["target"].values

logreg = Pipeline([
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=400, class_weight="balanced"))
])

rf = RandomForestClassifier(
    n_estimators=400, max_depth=5, min_samples_leaf=5,
    class_weight="balanced_subsample", random_state=42
)

thr = cv_choose_threshold(logreg, X, y)
print(f"Chosen threshold: {thr:.3f}")


# ================= LOGISTIC REGRESSION =================
print("\n=== Logistic Regression ===")
cm_lr, y_true_lr, y_pred_lr = final_holdout_eval(logreg, X, y, threshold=thr)

cut = int(len(X) * 0.8)
probs_lr = logreg.fit(X[:cut], y[:cut]).predict_proba(X[cut:])[:, 1]

plot_confusion_matrix(cm_lr, "Logistic Regression - Confusion Matrix")
plot_predictions(df, y_true_lr, y_pred_lr, cut)
plot_roc_curve(y_true_lr, probs_lr, "Logistic Regression - ROC Curve")
plot_precision_recall(y_true_lr, probs_lr, "Logistic Regression - Precision-Recall Curve")


# ================= RANDOM FOREST =================
print("\n=== Random Forest ===")
rf.fit(X[:cut], y[:cut])
probs_rf = rf.predict_proba(X[cut:])[:, 1]
rf_pred = (probs_rf >= thr).astype(int)

cm_rf = confusion_matrix(y[cut:], rf_pred)
print(confusion_matrix(y[cut:], rf_pred))
print(classification_report(y[cut:], rf_pred))

plot_confusion_matrix(cm_rf, "Random Forest - Confusion Matrix")
plot_feature_importances(rf, feature_cols)
plot_predictions(df, y[cut:], rf_pred, cut)
plot_roc_curve(y[cut:], probs_rf, "Random Forest - ROC Curve")
plot_precision_recall(y[cut:], probs_rf, "Random Forest - Precision-Recall Curve")


# ================= PRICE INDICATORS =================
plot_price_with_sma(df)
plot_rsi(df)
