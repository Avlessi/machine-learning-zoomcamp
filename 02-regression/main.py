# hw_car_mpg.py
import io
import math
import json
import requests
import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

def prepare_X(df):
    # design matrix with bias term 
    X = df.values
    ones = np.ones(X.shape[0])
    return np.column_stack([ones, X])

def train_linear_regression(X, y, r=0.0):
    # w = (X^T X + r * I_*)^{-1} X^T y
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    reg[0, 0] = 0.0
    XTX_inv = np.linalg.inv(XTX + reg)
    w = XTX_inv.dot(X.T).dot(y)
    return w

def predict(X, w):
    return X.dot(w)

def split_train_val_test(df, seed=42):
    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()


def main():

    csv = requests.get(URL).text
    df = pd.read_csv(io.StringIO(csv))

    cols = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
    df = df[cols].copy()

    
    target = 'fuel_efficiency_mpg'
    skew = df[target].skew()
    p50, p95 = df[target].quantile([0.5, 0.95])
    long_tail = (skew > 0.8) or ((p95 - p50) / (p50 + 1e-9) > 0.4)

    
    print(f"skew(fuel_efficiency_mpg): {skew:.3f}")
    print(f"median: {p50:.3f}, 95th pct: {p95:.3f}")
    print(f"Long right tail? {'Yes' if long_tail else 'No'}")

    
    na_counts = df.isna().sum()
    q1_col = na_counts[na_counts > 0].sort_values(ascending=False).index.tolist()
    q1_answer = q1_col[0] if q1_col else "None"

    horsepower_median = float(df['horsepower'].median())

    
    print(f"Missing counts:\n{na_counts}")
    print(f"Answer: {q1_answer}")

    
    print(f"Median(horsepower) = {horsepower_median:.0f}")

    
    df_train, df_val, df_test = split_train_val_test(df, seed=42)

    
    features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']

    y_train = df_train[target].values
    y_val   = df_val[target].values
    y_test  = df_test[target].values

    
    # Fill NAs with 0
    train_zero = df_train[features].fillna(0.0)
    val_zero   = df_val[features].fillna(0.0)

    X_train_zero = prepare_X(train_zero)
    X_val_zero   = prepare_X(val_zero)

    w0 = train_linear_regression(X_train_zero, y_train, r=0.0)
    rmse_zero = rmse(y_val, predict(X_val_zero, w0))

    # B) Fill NAs with train-mean
    means = df_train[features].mean(numeric_only=True)
    train_mean = df_train[features].fillna(means)
    val_mean   = df_val[features].fillna(means)

    X_train_mean = prepare_X(train_mean)
    X_val_mean   = prepare_X(val_mean)

    w_mean = train_linear_regression(X_train_mean, y_train, r=0.0)
    rmse_mean = rmse(y_val, predict(X_val_mean, w_mean))

    better_q3 = "With mean" if rmse_mean < rmse_zero - 1e-12 else ("With 0" if rmse_zero < rmse_mean - 1e-12 else "Both are equally good")

    
    print(f"  RMSE (0)   = {round(rmse_zero, 2)}")
    print(f"  RMSE (mean)= {round(rmse_mean, 2)}")
    print(f"  Answer: {better_q3}")

    
    r_list = [0, 0.01, 0.1, 1, 5, 10, 100]
    scores = []
    for r in r_list:
        w = train_linear_regression(X_train_zero, y_train, r=r)
        s = rmse(y_val, predict(X_val_zero, w))
        scores.append((r, round(s, 2)))

    best_rmse = min(s for (_, s) in scores)
    
    best_r = min(r for (r, s) in scores if s == best_rmse)

    
    for r, s in scores:
        print(f"r={r:<6}  RMSE={s:.2f}")
    print(f"Best r (tie -> smallest r): {best_r}")

    
    seed_vals = list(range(10))
    val_scores = []
    for sd in seed_vals:
        tr, va, te = split_train_val_test(df, seed=sd)
        y_tr = tr[target].values
        y_va = va[target].values

        X_tr = prepare_X(tr[features].fillna(0.0))
        X_va = prepare_X(va[features].fillna(0.0))

        w = train_linear_regression(X_tr, y_tr, r=0.0)
        s = rmse(y_va, predict(X_va, w))
        val_scores.append(s)

    std_val = float(np.std(val_scores))

    
    print("RMSEs:", [round(s, 3) for s in val_scores])
    print(f"std = {round(std_val, 3)}")

    
    tr, va, te = split_train_val_test(df, seed=9)
    y_tr = tr[target].values
    y_va = va[target].values
    y_te = te[target].values

    X_tr = prepare_X(tr[features].fillna(0.0))
    X_va = prepare_X(va[features].fillna(0.0))
    X_te = prepare_X(te[features].fillna(0.0))

    # combine train + val
    X_full = np.vstack([X_tr, X_va])
    y_full = np.concatenate([y_tr, y_va])

    w = train_linear_regression(X_full, y_full, r=0.001)
    test_rmse = rmse(y_te, predict(X_te, w))

    print(f"Test rmse = {round(test_rmse, 3)}")