import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "target"


def load_arff(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.decode("utf-8")
        df[col] = df[col].replace("?", "-1")
        df[col] = pd.to_numeric(df[col])
    return df


def describe_features(df):
    bin_cols = [c for c in df.columns if c.endswith("_bin")]
    cat_cols = [c for c in df.columns if c.endswith("_cat")]
    remaining = [c for c in df.columns if c not in bin_cols + cat_cols + [TARGET]]
    calc_cols = [c for c in remaining if "calc" in c]
    non_calc = [c for c in remaining if c not in calc_cols]
    ordinal_cols = [c for c in non_calc if (df[c].dropna() % 1 == 0).all()]
    cont_cols = [c for c in non_calc if c not in ordinal_cols]
    return bin_cols, cat_cols, ordinal_cols, cont_cols, calc_cols


def prepare_features(df, drop_calc=True, drop_high_missing=True):
    bin_cols, cat_cols, ordinal_cols, cont_cols, calc_cols = describe_features(df)
    high_missing = [c for c in cat_cols if (df[c] == -1).mean() > 0.40]
    drop = []
    if drop_calc:
        drop += calc_cols
    if drop_high_missing:
        drop += high_missing
    feature_cols = [c for c in df.columns if c not in drop + [TARGET]]
    X = df[feature_cols].copy()
    y = df[TARGET].astype(int).copy()
    for c in feature_cols:
        if c not in bin_cols:
            X[c] = X[c].replace(-1, np.nan)
    return X, y, {
        "bin": bin_cols,
        "cat": [c for c in cat_cols if c in feature_cols],
        "ordinal": [c for c in ordinal_cols if c in feature_cols],
        "continuous": [c for c in cont_cols if c in feature_cols],
        "calc": calc_cols,
        "high_missing": high_missing,
        "dropped": drop,
    }


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
