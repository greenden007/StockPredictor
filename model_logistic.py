# model_logistic.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV_PATH = "processed_data/AAPL.csv"  # change to another ticker if you want

def read_prices(csv_path: str) -> pd.DataFrame:
    """
    Robust CSV reader for OHLCV data:
    - Auto-detects delimiter
    - Normalizes column names (strip/lower/replace spaces)
    - Finds the date column (date, datetime, or first column if unnamed)
    - Coerces numeric columns to numeric
    - Prefers 'adj close' when present; otherwise uses 'close'
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find file: {csv_path}")

    # auto-detect delimiter; avoid dtype guessing problems
    df = pd.read_csv(csv_path, sep=None, engine="python")

    # normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # find/standardize date column
    date_col = None
    for candidate in ["date", "datetime", "time", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # if first column looks like the date (often unnamed)
        date_col = df.columns[0]
    # parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"}).sort_values("date")

    # identify price/volume columns (support many variants)
    name_map = {c: c for c in df.columns}
    # common aliases
    aliases = {
        "open": ["open", "open_price"],
        "high": ["high", "high_price"],
        "low": ["low", "low_price"],
        "close": ["close", "close_price", "last"],
        "adj_close": ["adj_close", "adjusted_close", "adjclose", "adj._close"],
        "volume": ["volume", "vol"],
    }

    def find_col(key):
        for alias in aliases[key]:
            if alias in df.columns:
                return alias
        return None

    open_c = find_col("open")
    high_c = find_col("high")
    low_c = find_col("low")
    close_c = find_col("adj_close") or find_col("close")  # prefer adjusted close
    volume_c = find_col("volume")

    required = [close_c, volume_c]
    if any(c is None for c in required):
        raise ValueError(
            f"Missing required columns. Close? {close_c is not None}, Volume? {volume_c is not None}. "
            f"Available columns: {list(df.columns)}"
        )

    # coerce numerics (handles strings like '1,234.56')
    numeric_cols = [c for c in [open_c, high_c, low_c, close_c, volume_c] if c is not None]
    for c in numeric_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep a tidy frame
    keep_cols = ["date", close_c, volume_c]
    if open_c: keep_cols.append(open_c)
    if high_c: keep_cols.append(high_c)
    if low_c:  keep_cols.append(low_c)
    df = df[keep_cols].dropna()

    # rename to standard names we’ll use below
    rename_map = {close_c: "close", volume_c: "volume"}
    if open_c: rename_map[open_c] = "open"
    if high_c: rename_map[high_c] = "high"
    if low_c:  rename_map[low_c]  = "low"
    df = df.rename(columns=rename_map)

    return df.reset_index(drop=True)

# -------- load & engineer --------
df = read_prices(CSV_PATH)

# features/target
df["return"] = df["close"].pct_change()
df["target"] = (df["return"].shift(-1) > 0).astype(int)
df["ma5"] = df["close"].rolling(5).mean()
df["ma10"] = df["close"].rolling(10).mean()
df["volatility"] = df["return"].rolling(10).std()
df["volume_change"] = df["volume"].pct_change()

df = df.dropna(subset=["return", "ma5", "ma10", "volatility", "volume_change", "target"]).reset_index(drop=True)

features = ["return", "ma5", "ma10", "volatility", "volume_change"]
X = df[features]
y = df["target"]

# -------- split, scale, train --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

# -------- evaluate --------
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

tail = df.loc[X_test.index, ["date", "target"]].copy()
tail["predicted"] = y_pred
print("\nTail preview:")
print(tail.tail(10).to_string(index=False))
